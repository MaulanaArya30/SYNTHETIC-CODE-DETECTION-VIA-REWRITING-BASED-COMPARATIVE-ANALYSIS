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
            device_map="auto",  
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f'{settings.REWRITER_MODEL_NAME} loaded successfully.')

    def _make_prompt(self, code):
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
        CANDIDATES_PER_ATTEMPT = min(num_rewrites * 2, 8)
        all_rewrites = [[] for _ in range(len(code_list))]
        attempt_counts = [0] * len(code_list) 
        max_attempts = num_rewrites * 2
        needs_more = list(range(len(code_list)))

        while needs_more:
            needs_more = [
                i for i in needs_more
                if len(all_rewrites[i]) < num_rewrites and attempt_counts[i] < max_attempts
            ]
            if not needs_more:
                break

            for i in needs_more:
                attempt_counts[i] += 1

            #build batch from all samples that still need rewrites
            batch_originals = [code_list[i] for i in needs_more]
            batch_prompts = [self._make_prompt(c) for c in batch_originals]

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True,
            ).to("cuda")


            with torch.no_grad():
                outputs = self.model.generate(
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

            for batch_pos, sample_idx in enumerate(needs_more):
                original = code_list[sample_idx]
                seq_start = batch_pos * CANDIDATES_PER_ATTEMPT
                seq_end = seq_start + CANDIDATES_PER_ATTEMPT

                for seq in outputs[seq_start:seq_end]:
                    if len(all_rewrites[sample_idx]) >= num_rewrites:
                        break
                    decoded = self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                    cleaned = self._extract_code(decoded)
                    if cleaned and self._is_valid_rewrite(cleaned, original):
                        all_rewrites[sample_idx].append(cleaned)
                        print(f"  [OK] sample {sample_idx} ({len(all_rewrites[sample_idx])}/{num_rewrites} collected)")

            #recompute which samples still need more rewrites
            needs_more = [
                i for i in range(len(code_list))
                if len(all_rewrites[i]) < num_rewrites and attempt_counts[i] < max_attempts
            ]

        #final results
        result = []
        for i, sample_rewrites in enumerate(all_rewrites):
            original = code_list[i]
            if not sample_rewrites:
                print(f"Warning: No valid rewrites for sample {i} after {attempt_counts[i]} attempts. Skipping.")
                result.append(None)
            elif len(sample_rewrites) < num_rewrites:
                print(f"Warning: Only {len(sample_rewrites)}/{num_rewrites} for sample {i}. Padding.")
                while len(sample_rewrites) < num_rewrites:
                    sample_rewrites.append(sample_rewrites[0])
                result.append(sample_rewrites)
            else:
                result.append(sample_rewrites)

        return result
    