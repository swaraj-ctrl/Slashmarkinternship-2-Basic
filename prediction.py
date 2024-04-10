import pickle
import numpy as np


with open('final_model.sav', 'rb') as model_file:
    model = pickle.load(model_file)


def preprocess_text(text):
    preprocessed_text = text.lower()
    return preprocessed_text

def classify_text(input_text):
    preprocessed_text = preprocess_text(input_text)
    numerical_features = [len(preprocessed_text)]
    numerical_features = np.array(numerical_features).reshape(1, -1)
    
    predicted_class = model.predict(numerical_features)
    probability = model.predict_proba(numerical_features)[:, 1] if predicted_class == 1 else model.predict_proba(numerical_features)[:, 0]
    
    return predicted_class, probability


def main():
    input_text = input("Enter the news headline or text you want to verify: ")
    predicted_class, probability = classify_text(input_text)
    if predicted_class == 1:
        print("The news is classified as TRUE with a probability of {:.2f}%".format(probability * 100))
    else:
        print("The news is classified as FALSE with a probability of {:.2f}%".format((1 - probability) * 100))

if __name__ == "__main__":
    main()
