from flask import Flask, render_template, request
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import demoji  # Import the demoji library
import re
import string

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('randomforest.joblib')

# Load the TF-IDF vectorizer
vectorizer = joblib.load('vectorizer.joblib')

# Initialize the demoji library
demoji.download_codes()

@app.route('/')
def home():  # Add a colon at the end
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  # Add a colon at the end
def predict():  # Add a colon at the end
    if request.method == 'POST':  # Add a colon at the end
        text_input = request.form['text_input']

        # Clean and prepare the input text
        cleaned_text = remove_mult_spaces(clean_hashtags(strip_all_entities(clean_text_from_emojis(text_input))))

        # Vectorize the input text using the pre-trained vectorizer
        text_vectorized = vectorizer.transform([cleaned_text])

        # Make predictions using the pre-trained model
        prediction = model.predict(text_vectorized)[0]

        return render_template('result.html', prediction=prediction, text_input=text_input)

def clean_text_from_emojis(text):
    return demoji.replace(text, '')

# Remove punctuations, links, mentions, and \r\n new line characters
def strip_all_entities(text):
    text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower()  # remove \n and \r and lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)  # remove links and mentions
    text = re.sub(r'[^\x00-\x7f]', r'', text)  # remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    banned_list = string.punctuation + 'Ã' + '±' + 'ã' + '¼' + 'â' + '»' + '§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text

# Clean hashtags '#' Symbol
def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet))  # remove last hashtags
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet))  # remove hashtags symbol from words in the middle of the sentence
    return new_tweet2

# Remove multiple spaces
def remove_mult_spaces(text):
    return re.sub("\s\s+", " ", text)

if __name__ == '__main__':
    app.run(debug=True)  # Add a colon at the end

