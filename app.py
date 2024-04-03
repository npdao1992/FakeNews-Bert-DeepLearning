from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load the trained BERT model
model_path = 'saved_model2'
tokenizer = BertTokenizer.from_pretrained('vinai/phobert-base')
model = BertForSequenceClassification.from_pretrained(model_path)

# Function to classify news
def classify_news(title, content):
    input_text = title + " " + content
    inputs = tokenizer.encode_plus(input_text, add_special_tokens=True, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).tolist()
    return "Reliable" if predictions[0] == 1 else "Unreliable"

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for form submission
@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['title']
    content = request.form['content']
    result = classify_news(title, content)
    return render_template('result.html', title=title, content=content, result=result)

if __name__ == '__main__':
    app.run(debug=True)
