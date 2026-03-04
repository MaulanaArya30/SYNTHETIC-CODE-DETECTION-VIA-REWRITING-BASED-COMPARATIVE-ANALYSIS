from transformers import RobertaTokenizer, T5ForConditionalGeneration
import settings
import torch

class CodeRewriter:
    def __init__(self):
        print('Loading tokenizer and model (CodeT5)...')
        self.tokenizer = RobertaTokenizer.from_pretrained(settings.REWRITER_MODEL_NAME)
        self.model = T5ForConditionalGeneration.from_pretrained(settings.REWRITER_MODEL_NAME)
        self.model = self.model.to(settings.DEVICE)
        self.model.eval()
        print('Model (CodeT5) and tokenizer loaded successfully.')

    def _extract_code(self, text):
        if "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                code = parts[1]
                if code.strip().lower().startswith("python"):
                    code = code[6:]
                
                # CRITICAL: Check if the extracted block is empty after stripping.
                if code.strip():
                    return code.strip()
                # If block is empty, fall through to default return.
        
        # If no triple backticks found, or if the extracted block was empty, 
        # assume the whole thing is the code (or is junk) and return stripped text.
        # If the whole text is junk/empty, let the next layer filter it.
        text = text.strip()
        if not text:
            return None # Return None if the output is empty
        return text

    def generate_rewrites(self, code_snippet, num_rewrites):
        prompts = [self._make_prompt(code_snippet)]
        result = self._generate(prompts, num_rewrites, [code_snippet])
        return result[0]
    
    def generate_rewrites_batch(self, code_list, num_rewrites):
        prompts = [self._make_prompt(c) for c in code_list]
        return self._generate(prompts, num_rewrites, code_list)
    def _make_prompt(self, code):
        return f"""
### Code:
{code}
### Instruction:
Please explain the functionality of the 
given code, then rewrite it in a 
single markdown code block. No 
additional clarifications.
"""

    def _generate(self, prompts, num_rewrites, original_codes):
        """
        Returns: list[list[str]]
        Outer list = batch
        Inner list = rewrites per input
        """

        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(settings.DEVICE)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=512,
                num_beams=num_rewrites * 2,
                num_return_sequences=num_rewrites,
                do_sample=True,
                top_p=0.95,
                temperature=0.8
            )

        #outputs is shape: (batch_size * num_rewrites, sequence_len)
        batch_size = len(prompts)

        all_rewrites = []
        idx = 0

        for batch_idx in range(batch_size):
            sample_rewrites = []
            failed_rewrites = 0
            
            for rewrite_idx in range(num_rewrites):
                decoded = self.tokenizer.decode(outputs[idx], skip_special_tokens=True)
                cleaned_code = self._extract_code(decoded)
                
                # Only append if the code snippet is valid (not None/empty)
                if cleaned_code:
                    sample_rewrites.append(cleaned_code)
                else:
                    failed_rewrites += 1
                
                idx += 1
            
            # === CRITICAL FIX: Handle empty rewrite lists ===
            if not sample_rewrites:
                # If ALL rewrites failed for this sample, use the original code as fallback
                # This prevents empty lists that cause division by zero in the detector
                print(f"Warning: All {num_rewrites} rewrites failed for sample {batch_idx}. Using original code as fallback.")
                sample_rewrites = [original_codes[batch_idx]]
            elif failed_rewrites > 0:
                # Some rewrites failed, but we have at least one valid rewrite
                # Pad with the original code to maintain the expected count
                print(f"Warning: {failed_rewrites}/{num_rewrites} rewrites failed for sample {batch_idx}. Padding with original.")
                while len(sample_rewrites) < num_rewrites:
                    sample_rewrites.append(original_codes[batch_idx])
            
            all_rewrites.append(sample_rewrites)

        return all_rewrites