from transformers import AutoTokenizer, AutoModelForCausalLM
import settings
import torch


class CodeRewriter:
    def __init__(self):
        print(f'Loading tokenizer and model ({settings.REWRITER_MODEL_NAME})...')

        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.REWRITER_MODEL_NAME,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.REWRITER_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",   # loads directly to GPU, avoids double-memory spike from .to()
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f'{settings.REWRITER_MODEL_NAME} loaded successfully.')

    def _make_prompt(self, code):
        """Use the model's official chat template for best instruction following."""
        messages = [
            {
                "role": "user",
                "content": (
                    f"Please explain the functionality of the given code, "
                    f"then rewrite it in a single markdown code block. "
                    f"No additional clarifications.\n\n"
                    f"```python\n{code}\n```"
                )
            }
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _extract_code(self, text):
        """Extract code from triple-backtick block."""
        if "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                code = parts[1]
                first_line = code.split("\n")[0].strip().lower()
                if first_line in ("python", "py", ""):
                    code = "\n".join(code.split("\n")[1:])
                if code.strip():
                    return code.strip()
        text = text.strip()
        return text if text else None

    def _is_valid_rewrite(self, rewrite, original):
        """
        Sanity checks - reject outputs that would pollute similarity scores.
        Returns False if the rewrite is empty, too short, identical to the
        original, or a repetition loop.
        """
        if not rewrite or len(rewrite.strip()) < 20:
            return False
        # Reject if way too short compared to original
        if len(rewrite) < len(original) * 0.15:
            return False
        # Reject identical copies - cosine sim would always be 1.0
        if rewrite.strip() == original.strip():
            return False
        # Reject repetition loops
        for i in range(len(rewrite) - 15):
            fragment = rewrite[i:i + 15]
            if rewrite.count(fragment) >= 4:
                return False
        return True

    def generate_rewrites(self, code_snippet, num_rewrites):
        return self.generate_rewrites_batch([code_snippet], num_rewrites)[0]

    def generate_rewrites_batch(self, code_list, num_rewrites):
        return self._generate(code_list, num_rewrites)

    def _generate(self, code_list, num_rewrites):
        """
        Generate num_rewrites rewritten versions per code sample.

        All m rewrites use the same prompt - variation comes naturally from
        do_sample=True with temperature=0.9, matching Ye et al. (2025).

        Uses num_return_sequences to generate all m rewrites in a single
        model.generate() call per sample.

        Returns list[list[str] | None]:
          - list[str] of length num_rewrites if ALL rewrites passed validation
          - None if ANY rewrite failed - the evaluator's valid_indices filter
            skips this sample entirely, avoiding artificial 1.0 cosine scores
            from padding with the original code.
        """
        self.tokenizer.padding_side = "left"
        all_rewrites = []

        for batch_idx, original in enumerate(code_list):
            prompt = self._make_prompt(original)
            sample_rewrites = []
            failed = 0

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=False,
            ).to(settings.DEVICE)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=512,
                    num_return_sequences=num_rewrites,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.9,
                    repetition_penalty=1.3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # outputs shape: (num_rewrites, sequence_len)
            # slice off prompt tokens to get only the generated part
            prompt_len = inputs.input_ids.shape[1]

            for seq in outputs:
                decoded = self.tokenizer.decode(
                    seq[prompt_len:], skip_special_tokens=True
                )
                cleaned = self._extract_code(decoded)

                if cleaned and self._is_valid_rewrite(cleaned, original):
                    sample_rewrites.append(cleaned)
                else:
                    failed += 1

            # Skip instead of pad - any failure discards the whole sample
            if failed > 0:
                reason = "All" if not sample_rewrites else f"{failed}/{num_rewrites}"
                print(f"Warning: {reason} rewrites failed for sample {batch_idx}. Skipping.")
                all_rewrites.append(None)
            else:
                all_rewrites.append(sample_rewrites)

        return all_rewrites