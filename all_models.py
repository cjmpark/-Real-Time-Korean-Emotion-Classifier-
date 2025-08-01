import pandas as pd
from konlpy.tag import Mecab 
import torch
import torch.nn
from transformers import ElectraTokenizer, ElectraForSequenceClassification
import pickle
import joblib
import numpy as np


class Koelectra:
    def __init__(self):
        model_path = "koelectra/electra_model_dict.pth"
        self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
        self.model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-small-v3-discriminator", num_labels=5)

        self.device = torch.device("cpu")
        point = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(point)
        self.model.to(self.device)
        self.model.eval()
        self.label2emotion = {0:"분노", 1:"슬픔", 2:"행복", 3:"두려움", 4:"중립"}

    def predict_emotion(self,text):
        encoding = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=150)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze()
            pred_label = torch.argmax(probs).item()
        return self.label2emotion[pred_label]

class LightML:
    def __init__(self, vectorizer, model):
        self.vectorizer = joblib.load(vectorizer)
        self.model = joblib.load(model)
        self.mecab = Mecab('/opt/homebrew/lib/mecab/dic/mecab-ko-dic')
        self.label2emotion = {0:"분노", 1:"슬픔", 2:"행복", 3:"두려움", 4:"중립"}

    def predict_emotion(self, text):
        txt = " ".join(self.mecab.morphs(str(text))) 
        input  = self.vectorizer.transform([txt])
        pred = self.model.predict(input)[0] 
        return self.label2emotion[pred]

class StackedEnsemble:
    def __init__(self, vectorizer, svm_model, sgdc_model, mnb_model, final_model):
        self.vectorizer = joblib.load(vectorizer)
        self.mecab = Mecab('/opt/homebrew/lib/mecab/dic/mecab-ko-dic')

        svm_model = joblib.load(svm_model)
        
        sgdc_model = joblib.load(sgdc_model)
        
        mnb_model = joblib.load(mnb_model)
        
        self.meta_model = joblib.load(final_model)
        
        self.base_model = {"svm":svm_model,
                           "SGDC":sgdc_model,
                           "MNB":mnb_model}
    
        self.label2emotion = {0:"분노", 1:"슬픔", 2:"행복", 3:"두려움", 4:"중립"}

    
    def predict_emotion(self, text):
        txt = " ".join(self.mecab.morphs(str(text)))
        input  = self.vectorizer.transform([txt])
        features = [model.predict_proba(input) for model in self.base_model.values()]

        meta_test_x = np.hstack(features)
        pred = self.meta_model.predict(meta_test_x)[0]
        return self.label2emotion[pred]


        