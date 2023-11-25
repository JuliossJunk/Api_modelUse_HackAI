from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertModel
import torch


app = FastAPI()

MAX_LEN = 256


class Item(BaseModel):
    body: str

class CustomClassifier(torch.nn.Module):
    def __init__(self, num_labels):
        super(CustomClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(last_hidden_state)
        logits = self.classifier(pooled_output)
        return logits


checkpoint = torch.load('model_e.pth', map_location=torch.device('cpu'))
model = CustomClassifier(num_labels=len(checkpoint['label_encoder'].classes_))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

label_encoder = checkpoint['label_encoder']

@app.post("/predict")
async def predict(item: Item):
    text = item.body
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    with torch.no_grad():
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        outputs = model(input_ids, attention_mask)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    predicted_category_index = torch.argmax(probabilities).item()
    predicted_category = label_encoder.classes_[predicted_category_index]

    return {"predicted_category": predicted_category}

