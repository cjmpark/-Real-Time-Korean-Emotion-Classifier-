import all_models  
import time

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

model_name = input("Choose Model: Koelecra(1), SVM(2), Stacked Ensemble(3)\n").strip().lower()


if model_name in ["koelectra", "1"]:
    model = all_models.Koelectra()
elif model_name in ["svm","2"]:
    model = all_models.LightML("svm/setAB_vectorizer.pkl", "svm/svm_setAB_0.8.pkl")
elif model_name in ["stacking", "3"]:
    model = all_models.StackedEnsemble(
        "stacking/setA_vector.pkl",
        "stacking/svm_model.pkl",
        "stacking/SGDC_model.pkl",
        "stacking/MNB_model.pkl",
        "stacking/meta_model.pkl"
    )
else:
    raise ValueError(f"Model '{model_name}' is not recognized.")

print(f"Real-Time Emotion Tracker started using [{model_name.upper()}]\n")
while True:
    user_input = input("Input: ")
    if user_input.strip().lower() == "q":
        print("Exiting real-time tracker.")
        break

    start = time.time()
    emotion = model.predict_emotion(user_input)
    elapsed = time.time() - start

    print(f"Model: {model_name.upper()}")
    print(f"Time: {elapsed:.3f} sec\n")
    print(f"Emotion: {emotion}")
