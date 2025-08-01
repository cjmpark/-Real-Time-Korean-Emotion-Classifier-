# Real-Time Korean Emotion Classifier  

---
### 📝 Description

**ENGLISH**  
This project is a real-time Korean emotion classifier that analyzes text input and predicts one of five emotions:  
→ **Anger**, **Sadness**, **Happiness**, **Fear**, **Neutral**.  
It supports multiple model backends including:
- KoELECTRA (transformer-based from Koelectra)
- SVM
- Stacked Ensemble (SVM + SGDC + Naive Bayes)

All of the models are trained from AI-Hub (https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=271, https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=86) and custom made datasets. Training scripts and datasets are not included, only final model and testing script is. 

---

### 🔧 Installation

```bash
# Create a new conda environment
conda create -n demo python=3.10
conda activate demo
pip install -r requirements.txt
```
