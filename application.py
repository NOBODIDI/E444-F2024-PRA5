from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

application = Flask(__name__)

@application.route("/")
def index():
    return "ECE444 Fake News Classifier is running!"

def load_model():
    with open('basic_classifier.pkl', 'rb') as fid:
        loaded_model = pickle.load(fid)
    
    with open('count_vectorizer.pkl', 'rb') as vd:
        vectorizer = pickle.load(vd)
    
    return loaded_model, vectorizer

@application.route("/classify", methods=["POST"])
def classify_text():
    try:
        model, vectorizer = load_model()
        data = request.get_json()

        if 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        text = data['text']
        vectorized_text = vectorizer.transform([text])
        prediction = model.predict(vectorized_text)[0]

        result = prediction == 1
        print(f"Text: {text}, Classified as: {'True' if result else 'False'}")

        return jsonify({"classification": result}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    application.run(port=5000, debug=True)
