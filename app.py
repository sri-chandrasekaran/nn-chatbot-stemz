from flask import Flask, render_template, request, jsonify
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import textdistance
from spellchecker import SpellChecker

nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

ignore_letters = ['?', '!', '.', ',']

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    # sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word not in ignore_letters]
    return sentence_words

def correct_spelling(sentence_words):
    corrected_words = []
    for word in sentence_words:
        corrected_word = spell.correction(word)
        corrected_words.append(corrected_word)
    return corrected_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    print(f"Bag of words: {bow}")  # Debugging output
    res = model.predict(np.array([bow]))[0]
    print(f"Model prediction: {res}")  # Debugging output
    ERROR_THRESHOLD = 0.1  # Lowered threshold
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    print(f"Filtered results: {results}")  # Debugging output

    if results and results[0][1] < 0.7:
        results = []

    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# def get_response(ints, intents_json):
#     tag = ints[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if i['tag'] == tag:
#             result = random.choice(i['responses'])
#             break
#     return result

# def get_closest_match(ints):
#     if not ints:
#         return None

#     max_similarity = 0
#     closest_match = None
#     for intent in intents['intents']:
#         for pattern in intent['patterns']:
#             similarity = textdistance.jaccard.normalized_similarity(ints[0]['intent'], pattern)
#             if similarity > max_similarity:
#                 max_similarity = similarity
#                 closest_match = intent['tag']

#     if max_similarity < 0.5:
#         return None
#     return closest_match


# def get_response(ints, intents_json):
#     if not ints:
#         print("No intent matched. Returning default response.")
#         return "Sorry, I didn't understand that. Please send us a message using our Contact page."
#     tag = ints[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if i['tag'] == tag:
#             result = random.choice(i['responses'])
#             break
#     return result

def get_closest_match(sentence):
    highest_similarity = 0
    closest_intent = None

    sentence_words = clean_up_sentence(sentence)
    corrected_sentence = correct_spelling(sentence_words)
    corrected_sentence = [word for word in corrected_sentence if word is not None]
    corrected_sentence_str = ' '.join(corrected_sentence)

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            similarity = textdistance.jaccard.normalized_similarity(corrected_sentence_str, pattern)
            if similarity > highest_similarity:
                highest_similarity = similarity
                closest_intent = intent['tag']

    if highest_similarity < 0.5:  # Adjust threshold as needed
        return None
    return closest_intent

def get_response(ints, intents_json, sentence):
    if not ints:
        closest_match = get_closest_match(sentence)
        if closest_match:
            for intent in intents_json['intents']:
                if intent['tag'] == closest_match:
                    return random.choice(intent['responses'])
        return "Sorry, I didn't understand that. Please send us a message using our Contact page and we will get back to you."
    tag = ints[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that. Please send us a message using our Contact page and we will get back to you."
    

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    ints = predict_class(message)
    print(f"Predicted intents: {ints}")
    res = get_response(ints, intents, message)
    return jsonify({'response': res})

if __name__ == "__main__":
    app.run(debug=True)
