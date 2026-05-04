import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, T5ForConditionalGeneration, RobertaModel
import settings
import os

class CodeDetector:
    def __init__(self, model_type, use_simcse=True):
        self.model_type = model_type
        self.use_simcse = use_simcse

        if model_type == "codet5":
            if use_simcse:
                model_path = settings.CODET5_SIMCSE_PATH
                print(f"Loading CodeT5-SimCSE from {model_path}...")
            else:
                model_path = settings.CODET5_MODEL_NAME
                print(f"Loading BASE CodeT5 from {model_path}...")
            
            #check if SimCSE ver exists
            if use_simcse and not os.path.exists(model_path):
                print(f" SimCSE model not found at {model_path}")
                print(f"   Falling back to base model: {settings.CODET5_MODEL_NAME}")
                model_path = settings.CODET5_MODEL_NAME
                self.use_simcse = False
            
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.encoder = self.model.encoder

        elif model_type == "graphcodebert":
            if use_simcse:
                model_path = settings.GCB_SIMCSE_PATH
                print(f"Loading GraphCodeBERT-SimCSE from {model_path}...")
            else:
                model_path = settings.GCB_MODEL_NAME
                print(f"Loading BASE GraphCodeBERT from {model_path}...")
            
            #check if SimCSE ver exists
            if use_simcse and not os.path.exists(model_path):
                print(f"SimCSE model not found at {model_path}")
                print(f"Falling back to base model: {settings.GCB_MODEL_NAME}")
                model_path = settings.GCB_MODEL_NAME
                self.use_simcse = False
            
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
            self.model = RobertaModel.from_pretrained(model_path)
            self.encoder = self.model

        else:
            raise ValueError("Invalid model_type. Choose 'codet5' or 'graphcodebert'")

        self.encoder = self.encoder.to(settings.DEVICE)
        self.encoder.eval()
        
        model_type_str = f"{model_type.upper()} ({'SimCSE' if self.use_simcse else 'BASE'})"
        print(f"{model_type_str} model loaded successfully")

    #single emb 
    @torch.no_grad()
    def get_embedding(self, code):
        return self.get_embedding_batch([code])[0]

    #batch emb
    @torch.no_grad()
    def get_embedding_batch(self, code_list):
        inputs = self.tokenizer(
            code_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(settings.DEVICE)

        outputs = self.encoder(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )

        #codet5 uses mean pooling
        if self.model_type == "codet5":
            embeddings = outputs.last_hidden_state.mean(dim=1)
        #graphcodebert uses CLS token
        else:
            embeddings = outputs.last_hidden_state[:, 0]
        #normalize due to cosine similarity usage
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    #single code sample score
    def get_detection_score(self, original, rewritten_list):
        return self.get_detection_score_batch([original], [rewritten_list])[0]

    #batch score
    def get_detection_score_batch(self, originals, rewritten_lists):
        if not all(rewritten_lists):
            raise ValueError(
                "Some rewrite lists are empty! Check rewriter output. "
                "All samples must have at least one rewrite."
            )

        #original embeddings
        orig_embeds = self.get_embedding_batch(originals)

        flat_rewrites = [r for group in rewritten_lists for r in group]
        rewrite_embeds = self.get_embedding_batch(flat_rewrites)

        #regroup per sample
        rewrites_per_sample = [len(group) for group in rewritten_lists]
        grouped_embeds = []
        start_idx = 0
        for count in rewrites_per_sample:
            grouped_embeds.append(rewrite_embeds[start_idx:start_idx + count])
            start_idx += count

        scores = []
        for i, rew_emb in enumerate(grouped_embeds):
            sims = F.cosine_similarity(
                orig_embeds[i].unsqueeze(0),
                rew_emb,
                dim=1
            )
            
            avg_score = float(sims.mean())
            scores.append(avg_score)

        return scores
