from transformers import RobertaTokenizer, T5ForConditionalGeneration
from src import settings
import torch

class CodeRewriter:
    def __init__(self):
        print('Loading tokenizer and model (CodeT5)...')
        self.tokenizer = RobertaTokenizer.from_pretrained(settings.REWRITER_MODEL_NAME)
        self.model = T5ForConditionalGeneration.from_pretrained(settings.REWRITER_MODEL_NAME)
        self.model = self.model.to(settings.DEVICE)
        self.model.eval()
        print('Model (CodeT5) and tokenizer loaded successfully.')

    def generate_rewrites(self, code_snippet, num_rewrites):
        prompt = f"""
### Code:
{code_snippet}

### Instructions:
Please explain the functionality of the
give code, then rewrite it in a 
single markdown code block. No
additional clarifications.
"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(settings.DEVICE)

        #generate rewrites
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs.input_ids,
                max_length=512,
                num_beams=num_rewrites * 2,
                num_return_sequences=num_rewrites,
                early_stopping=True,
                do_sample=True, 
                top_p=0.95,
                temperature=0.8
            )

        rewritten_codes = []
        for out_id in output_ids:
            rewritten_text = self.tokenizer.deode(
                out_id,
                skip_special_tokens=True,
            )

            #extract only the code
            if "```" in rewritten_text:
                parts = rewritten_text.split("```")
                if len(parts) > 1 :
                    code_block = parts[1]
                    if code_block.lower().startswith("python"):
                        code_block = code_block[6:]
                    rewritten_codes.append(code_block.strip())
                else:
                    rewritten_codes.append(rewritten_text)
            else:
                rewritten_codes.append(rewritten_text)
            
        return rewritten_codes

            