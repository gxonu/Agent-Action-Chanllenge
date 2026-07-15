"""mmbert(ModernBertForSequenceClassification) pooled 벡터 + logits 추출.
규중 T5Gemma용 model.encoder/attn/classifier 를 대체 (mean pooling + head)."""
import torch
@torch.no_grad()
def mmbert_pooled_logits(model, enc):
    out = model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    H = out.last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).to(H.dtype)
    pooled_hidden = (H * mask).sum(1) / mask.sum(1).clamp(min=1e-6)   # classifier_pooling='mean'
    pooled = model.head(pooled_hidden)          # ModernBertPredictionHead
    logits = model.classifier(pooled)
    return pooled, logits
