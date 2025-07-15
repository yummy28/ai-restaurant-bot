import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import joblib

model_path = "../distilbert_query_classifier"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
le = joblib.load(f"{model_path}/label_encoder.pkl")

device = torch.device("cpu")
model.to(device)
model.eval()

def classify_query(query: str) -> str:
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding='max_length', max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class_id = outputs.logits.argmax(dim=-1).item()

    return le.inverse_transform([predicted_class_id])[0]
