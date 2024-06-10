import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torchvision.transforms as T
import re
from ultralytics import YOLO
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words("russian"))

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

#BERT
class BERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained('cointegrated/rubert-tiny-toxicity')
        self.bert.classifier = nn.Linear(312, 312)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.linear = nn.Sequential(
            nn.Linear(312, 128),
            nn.Sigmoid(),
            nn.Dropout(),
            nn.Linear(128, 1)
        )

    def forward(self, x, attention_mask=None):
        bert_out = self.bert(x, attention_mask=attention_mask).logits
        out = self.linear(bert_out).squeeze(1)
        return out

def get_absolute_path(relative_path):
    script_dir = os.path.dirname(__file__)
    return os.path.join(script_dir, relative_path)

def load_model():
    '''
    Returns BERT model with pre-trained weights
    '''
    model = BERTClassifier()
    state_dict = torch.load(get_absolute_path('model_weights_new.pth'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_tokenizer():
    return AutoTokenizer.from_pretrained('cointegrated/rubert-tiny-toxicity')

def clean_text(text):
    # Удаление всего, что не является буквами или знаками препинания
    clean_pattern = re.compile(r'[^a-zA-Zа-яА-ЯёЁ0-9.,!?;:\s]')
    text = clean_pattern.sub('', text)
    url_pattern = re.compile(r'http\S+|www\S+|https\S+')
    text = url_pattern.sub(r'', text)
    text = re.sub("\s+", " ", text)
    splitted_text = [word for word in text.split() if word not in stop_words]
    text = " ".join(splitted_text)
    return text

#YOLOv8
def load_yolo():
    # yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    # yolo.eval()
    yolo = YOLO("yolov8n.pt")
    yolo.fuse()
    return yolo

def transform_image(img):
    '''
    Input: PIL img
    Returns: transformed image
    '''
    trnsfrms = T.Compose(
        [
            T.Resize((224, 224)), 
            T.CenterCrop(100),
            T.ToTensor(),
            T.Normalize(mean, std)
        ]
    )
    return trnsfrms(img)
