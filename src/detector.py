import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, T5ForConditionalGeneration, RobertaModel
import settings

class CodeDetector:
    def __init__(self, model_type):
        self.model_type = model_type

        if model_type == 'codet5':
            model_path = settings.CODET5_SIMCSE_PATH
            print(f'Loading Model (codet5-simcse) from {model_path}...')
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.encoder = self.model.encoder
        elif model_type == 'graphcodebert':
            model_path = settings.GCB_SIMCSE_PATH
            print(f'Loading Model (graphcodebert-simcse) from {model_path}...')
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
            self.model = RobertaModel.from_pretrained(model_path)
            self.encoder = self.model
        else:
            raise ValueError('Invalid model_tyoe...')
        
        self.encoder = self.encoder.to(settings.DEVICE)
        self.encoder.eval()
        print(f'Model {model_type} loaded...')

    @torch.no_grad() #disable gradient calculation
    def get_embedding(self, code_snippet):
        inputs = self.tokenizer(
            code_snippet,
            return_tensors="pt",
            max_length=512,
            padding="max_length",
            truncation=True
        ).to(settings.DEVICE)
        
        outputs = self.encoder(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

        #get embedding
        if self.model_type == 'codet5':
            embedding = outputs.last_hidden_state.mean(dim=1)
        else: #graphcodebert
            embedding = outputs.last_hidden_state[:, 0]
        return embedding
    
    def get_detection_score(self, original_code, rewritten_codes):
        if not rewritten_codes:
            return 0.0
        
        original_embedding = self.get_embedding(original_code)

        total_similarity = 0
        for rewritten_code in rewritten_codes:
            rewritten_embedding = self.get_embedding(rewritten_code)

            #cosine similarity
            similarity = F.cosine_similarity(original_embedding, rewritten_embedding).item()
            total_similarity += similarity
        
        #average the score
        return total_similarity / len(rewritten_codes)