import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch.nn.functional as F

# Load model and tokenizer
model_path = "./model"
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()

    label = "positive" if predicted_class == 1 else "negative"
    return label, confidence

# Test predictions
examples = [
    "This movie was absolutely fantastic!",
    "I hated every second of it. Terrible.",
    "It was okay, not the best but not the worst either.",
    "Brilliant acting and a great story!",
    "Waste of time. Do not recommend."
]

for text in examples:
    label, conf = predict_sentiment(text)
    print(f"Text: {text}\nSentiment: {label} ({conf:.2f} confidence)\n")
