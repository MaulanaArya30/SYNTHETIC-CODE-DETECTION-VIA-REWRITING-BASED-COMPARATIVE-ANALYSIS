from transformers import AutoTokenizer, AutoModelForCausalLM
import settings
import torch


class CodeRewriter:
    def __init__(self):
        print('Loading tokenizer and model (DeepSeek-Coder-6.7B-Instruct)...')

        # DeepSeek-Coder uses AutoTokenizer / AutoModelForCausalLM
        # The model is a causal (decoder-only) LM, unlike CodeT5 which was encoder-decoder.
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.REWRITER_MODEL_NAME,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.REWRITER_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,   # fp16 halves VRAM usage (~14GB → ~8GB)
        )
        self.model = self.model.to(settings.DEVICE)
        self.model.eval()

        # DeepSeek tokenizer may not set a pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print('DeepSeek-Coder-6.7B-Instruct loaded successfully.')

    # ── Prompt format ─────────────────────────────────────────────────────────
    # DeepSeek-Coder-Instruct expects the Alpaca-style chat template:
    #   ### Instruction:\n{instruction}\n### Response:\n
    # This is the same logical prompt as before, just wrapped correctly.
    def _make_prompt(self, code):
        return (
            f"### Instruction:\n"
            f"Please explain the functionality of the given code, "
            f"then rewrite it in a single markdown code block. "
            f"No additional clarifications.\n\n"
            f"### Code:\n{code}\n\n"
            f"### Response:\n"
        )

    def _extract_code(self, text):
        """Extract code from triple-backtick block, same logic as before."""
        if "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                code = parts[1]
                # Strip language tag (e.g. ```python)
                first_line = code.split("\n")[0].strip().lower()
                if first_line in ("python", "py", ""):
                    code = "\n".join(code.split("\n")[1:])
                if code.strip():
                    return code.strip()

        text = text.strip()
        return text if text else None

    def _is_valid_rewrite(self, rewrite, original):
        """
        Basic sanity checks to catch corrupted / degenerate outputs.
        Returns False if the rewrite looks garbled.
        """
        if not rewrite or len(rewrite.strip()) < 20:
            return False

        # Reject if way too short compared to original
        if len(rewrite) < len(original) * 0.15:
            return False

        # Reject if repetition loop detected:
        # check if any 15-char fragment appears 4+ times
        for i in range(len(rewrite) - 15):
            fragment = rewrite[i:i + 15]
            if rewrite.count(fragment) >= 4:
                return False

        return True

    def generate_rewrites(self, code_snippet, num_rewrites):
        result = self.generate_rewrites_batch([code_snippet], num_rewrites)
        return result[0]

    def generate_rewrites_batch(self, code_list, num_rewrites):
        prompts = [self._make_prompt(c) for c in code_list]
        return self._generate(prompts, num_rewrites, code_list)

    def _generate(self, prompts, num_rewrites, original_codes):
        """
        Returns: list[list[str]]
        Outer list = one entry per input code sample
        Inner list = num_rewrites rewritten versions

        Key differences from the CodeT5 version:
        - DeepSeek is a CAUSAL (decoder-only) model, so we use left-padding
          and generate continuations rather than seq2seq outputs.
        - We generate each sample individually (or in small batches) because
          causal LMs with left-padding and varying prompt lengths are more
          reliably handled one at a time.
        - We generate num_rewrites sequences per sample using do_sample=True.
        """
        # Switch to left padding for causal generation
        self.tokenizer.padding_side = "left"

        all_rewrites = []

        for batch_idx, (prompt, original) in enumerate(zip(prompts, original_codes)):
            sample_rewrites = []
            failed = 0

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,          # longer context for DeepSeek
                padding=False,
            ).to(settings.DEVICE)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=512,       # only count NEW tokens generated
                    num_return_sequences=num_rewrites,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.8,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Slice off the prompt tokens — we only want the generated part
            prompt_len = inputs.input_ids.shape[1]

            for seq in outputs:
                generated_tokens = seq[prompt_len:]
                decoded = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                )
                cleaned = self._extract_code(decoded)

                if cleaned and self._is_valid_rewrite(cleaned, original):
                    sample_rewrites.append(cleaned)
                else:
                    failed += 1

            # Fallback handling — same logic as before
            if not sample_rewrites:
                print(f"Warning: All {num_rewrites} rewrites failed for sample {batch_idx}. "
                      f"Using original as fallback.")
                sample_rewrites = [original] * num_rewrites
            elif failed > 0:
                print(f"Warning: {failed}/{num_rewrites} rewrites failed for sample {batch_idx}. "
                      f"Padding with original.")
                while len(sample_rewrites) < num_rewrites:
                    sample_rewrites.append(original)

            all_rewrites.append(sample_rewrites)

        return all_rewrites