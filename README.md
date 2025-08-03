# Real-Time Korean Emotion Classifier  

---
### Description

**ENGLISH**  
This project is a real-time Korean emotion classifier that analyzes text input and predicts one of five emotions:  
→ **Anger**, **Sadness**, **Happiness**, **Fear**, **Neutral**. 

It supports multiple model backends including:

- **KoELECTRA** (Transformer-based model from [KoELECTRA](https://github.com/monologg/KoELECTRA))
- **SVM**
- **Stacked Ensemble**  
  (Base: SVM + SGDClassifier + MultinomialNB | Meta: Logistic Regression)

All models are trained on:
- [AI Hub Korean Emotion Datasets](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=271)
- [AI Hub Dialogue Emotion Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=86)
- Plus custom-augmented corpora

> **Note**: Only trained model files and real-time testing scripts are included.  
> Training scripts and raw datasets are **not** provided.

---

### Installation

```bash
# Create a new conda environment
conda create -n demo python=3.10
conda activate demo
pip install -r requirements.txt
```

> If Mecab/Konlpy does not work, follow [KoNLPy Guide](https://konlpy.org/ko/v0.4.0/install/) for more instructions. 

---

### How to run:
- To run all three models
```python
python all_model_test.py
```
- To run single model:
```python
python single_model_test.py
```

---

### Results + Short architecture
- Other models were trained also, and the best three were Koelectra, SVM, and Stacked Ensemble.


---

### License

This repository is intended for research and internal development use only.
Please use it solely for academic or evaluation purposes.

---

### Acknowledgements

- https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=271
- https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=86
- https://github.com/monologg/KoELECTRA
- https://arxiv.org/pdf/1408.5882
- https://konlpy.org/ko/v0.4.0/install/
  








