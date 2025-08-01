import all_models  
import time

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


koelectra = all_models.Koelectra()
svm = all_models.LightML("svm/setAB_vectorizer.pkl", "svm/svm_setAB_0.8.pkl")
stacked_ensemble = all_models.StackedEnsemble("stacking/setA_vector.pkl",
                                              "stacking/svm_model.pkl",
                                              "stacking/SGDC_model.pkl",
                                              "stacking/MNB_model.pkl",
                                              "stacking/meta_model.pkl")

models = {"koelectra": koelectra, 
          "svm": svm,
          "stacked_ensemble":stacked_ensemble}

print(f"Real-Time Emotion Tracker started using \n")
while True:
    user_input = input("Input: ")
    if user_input.strip().lower() == "q":
        print("Exiting real-time tracker.")
        break
    for name, model in models.items():
        start = time.time()
        emotion = model.predict_emotion(user_input)
        elapsed = time.time() - start

        print(f"Model: {name.upper()}")
        print(f"Time: {elapsed:.3f} sec")
        print(f"Emotion: {emotion}\n")
    
