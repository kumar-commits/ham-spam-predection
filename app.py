
import pickle
import nltk
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import streamlit as st
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer





ps = PorterStemmer()
Tfidf = pickle.load(open('vectorizer4.pkl','rb'))
model = pickle.load(open('model4.pkl','rb'))


def transform_text(text):
      text = text.lower()
      text = nltk.word_tokenize(text)

      y = []
      for i in text:
        if i.isalnum():
            y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            y.append(ps.stem(i))

        return " ".join(y)






st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = Tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    
    # 4. Display
    if result == 1:
        st.header("SPAM")
    else:
        st.header("HAM")
