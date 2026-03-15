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
                    f"Rewrite the following Python code. "
                    f"Return ONLY the rewritten code in a single markdown code block. "
                    f"Do not explain, do not add comments.\n\n"
                    f"{code}"
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
        if not rewrite or len(rewrite.strip()) < 10:
            print(f"  [FAIL] Too short or empty: len={len(rewrite.strip()) if rewrite else 0}")
            return False
        if len(rewrite) < len(original) * 0.10:
            print(f"  [FAIL] Too short vs original: {len(rewrite)} < {len(original) * 0.10:.0f}")
            return False
        if rewrite.strip() == original.strip():
            print(f"  [FAIL] Identical copy")
            return False
        for i in range(len(rewrite) - 20):
            fragment = rewrite[i:i + 20]
            if rewrite.count(fragment) >= 6:
                print(f"  [FAIL] Repetition loop detected")
                return False
        return True

    def generate_rewrites(self, code_snippet, num_rewrites):
        return self.generate_rewrites_batch([code_snippet], num_rewrites)[0]

    def generate_rewrites_batch(self, code_list, num_rewrites):
        return self._generate(code_list, num_rewrites)





    def _generate(self, code_list, num_rewrites):
        self.tokenizer.padding_side = "left"
        all_rewrites = []

        for batch_idx, original in enumerate(code_list):
            prompt = self._make_prompt(original)
            sample_rewrites = []
            max_attempts = num_rewrites * 2
            attempts = 0

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=False,
            ).to(settings.DEVICE)

            CANDIDATES_PER_ATTEMPT = min(num_rewrites * 2, 8)

            while len(sample_rewrites) < num_rewrites and attempts < max_attempts:
                attempts += 1

                with torch.no_grad():
                    output = self.model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=256,
                        num_return_sequences=CANDIDATES_PER_ATTEMPT,
                        do_sample=True,
                        top_p=0.95,
                        temperature=0.9,
                        repetition_penalty=1.5,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                prompt_len = inputs.input_ids.shape[1]
                for seq in output:
                    if len(sample_rewrites) >= num_rewrites:
                        break
                    decoded = self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                    cleaned = self._extract_code(decoded)
                    if cleaned and self._is_valid_rewrite(cleaned, original):
                        sample_rewrites.append(cleaned)
                        print(f"  [OK] attempt {attempts}/{max_attempts} succeeded for sample {batch_idx} ({len(sample_rewrites)}/{num_rewrites} collected)")

                    else:
                        print(f"  [RETRY] attempt {attempts}/{max_attempts} failed for sample {batch_idx}")

            if not sample_rewrites:
                # No valid rewrites at all after all attempts — skip entirely
                print(f"Warning: No valid rewrites for sample {batch_idx} after {max_attempts} attempts. Skipping.")
                all_rewrites.append(None)
            elif len(sample_rewrites) < num_rewrites:
                # Got some but not enough — pad with valid rewrites
                print(f"Warning: Only {len(sample_rewrites)}/{num_rewrites} valid rewrites for sample {batch_idx}. Padding with valid rewrites.")
                while len(sample_rewrites) < num_rewrites:
                    sample_rewrites.append(sample_rewrites[0])
                all_rewrites.append(sample_rewrites)
            else:
                # All good
                all_rewrites.append(sample_rewrites)

        return all_rewrites
    # def _generate(self, code_list, num_rewrites):
    #     """
    #     Generate num_rewrites rewritten versions per code sample.

    #     All m rewrites use the same prompt - variation comes naturally from
    #     do_sample=True with temperature=0.9, matching Ye et al. (2025).

    #     Uses num_return_sequences to generate all m rewrites in a single
    #     model.generate() call per sample.

    #     Returns list[list[str] | None]:
    #       - list[str] of length num_rewrites if ALL rewrites passed validation
    #       - None if ANY rewrite failed - the evaluator's valid_indices filter
    #         skips this sample entirely, avoiding artificial 1.0 cosine scores
    #         from padding with the original code.
    #     """
    #     self.tokenizer.padding_side = "left"
    #     all_rewrites = []

    #     for batch_idx, original in enumerate(code_list):
    #         prompt = self._make_prompt(original)
    #         sample_rewrites = []
    #         failed = 0

    #         inputs = self.tokenizer(
    #             prompt,
    #             return_tensors="pt",
    #             truncation=True,
    #             max_length=1024,
    #             padding=False,
    #         ).to(settings.DEVICE)

    #         with torch.no_grad():
    #             outputs = self.model.generate(
    #                 input_ids=inputs.input_ids,
    #                 attention_mask=inputs.attention_mask,
    #                 max_new_tokens=1024,
    #                 num_return_sequences=num_rewrites,
    #                 do_sample=True,
    #                 top_p=0.95,
    #                 temperature=0.9,
    #                 repetition_penalty=1.5,
    #                 pad_token_id=self.tokenizer.pad_token_id,
    #                 eos_token_id=self.tokenizer.eos_token_id,
    #             )

    #         # outputs shape: (num_rewrites, sequence_len)
    #         # slice off prompt tokens to get only the generated part
    #         prompt_len = inputs.input_ids.shape[1]

    #         for seq in outputs:
    #             decoded = self.tokenizer.decode(
    #                 seq[prompt_len:], skip_special_tokens=True
    #             )
    #             cleaned = self._extract_code(decoded)

    #             if cleaned and self._is_valid_rewrite(cleaned, original):
    #                 sample_rewrites.append(cleaned)
    #             else:
    #                 failed += 1

    #         # Skip instead of pad - any failure discards the whole sample
    #         if not sample_rewrites:
    #             # ALL failed — genuinely no valid rewrite, skip
    #             print(f"Warning: All {num_rewrites} rewrites failed for sample {batch_idx}. Skipping.")
    #             all_rewrites.append(None)
    #         elif failed > 0:
    #             # SOME failed — pad with the valid ones we do have
    #             print(f"Warning: {failed}/{num_rewrites} rewrites failed for sample {batch_idx}. Padding with valid rewrites.")
    #             while len(sample_rewrites) < num_rewrites:
    #                 sample_rewrites.append(sample_rewrites[0])  # pad with first valid rewrite, not original
    #             all_rewrites.append(sample_rewrites)
    #         else:
    #             all_rewrites.append(sample_rewrites)

    #     return all_rewrites
