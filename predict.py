import joblib
import sys

def predict(text, model_path):
    model = joblib.load(model_path)
    prediction = model.predict([text])[0]
    result = "🟢 Real" if prediction == 1 else "🔴 Fake"
    print(f"Prediction: {result}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py <dataset> <your_text>")
        print("Example: python predict.py isot \"The president signed a new bill today.\"")
        sys.exit(1)

    dataset = sys.argv[1]
    user_text = " ".join(sys.argv[2:])

    if dataset == "isot":
        model_file = "model/isot_model.pkl"
    elif dataset == "liar":
        model_file = "model/liar_model.pkl"
    else:
        print("❌ Unknown dataset. Use 'isot' or 'liar'.")
        sys.exit(1)

    predict(user_text, model_file)
