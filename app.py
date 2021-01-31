import streamlit as st
import pandas as pd
import os
import gzip, pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

stop_words =stopwords.words('german')

def cleaning(data):
    
    #1. Tokenize
    text_tokens = word_tokenize(data)
    
    #2. Remove Puncs
    tokens_without_punc = [w for w in text_tokens if w.isalpha()]
    
    #3. Removing Stopwords
    tokens_without_sw = [t for t in tokens_without_punc if t.lower() not in stop_words]
    
    #4. lemma
    text_cleaned = [WordNetLemmatizer().lemmatize(t) for t in tokens_without_sw]
    
    #joining
    return " ".join(text_cleaned)


with gzip.open('vectorizer.pklz', 'rb') as ifp:
    vector = pickle.load(ifp)
with gzip.open('vectorizer_emding.pklz', 'rb') as ifp:
    vectorizer_emding = pickle.load(ifp)
with gzip.open('vector_parag.pklz', 'rb') as ifp:
    vectorizer_parag = pickle.load(ifp)

def main():
#title
    st.title("Text Analyzer")
    st.subheader("Context and Similarity Analyser")
    st.markdown("""
    	#### Description
    	+ This is a Natural Language Processing(NLP) Based App useful for analysing 
    	the context and the similarity between sentences of Frau Frau Merkel and Herr Steinmeir
    	""")



    if st.checkbox("Show Logistic Analysis for Long Textes"):
        with gzip.open('model_parag.pklz', 'rb') as ifp:
            model_log_parag = pickle.load(ifp)
        st.subheader("Text Analysis")
        message_log_parag = st.text_area("Enter Your Parag    ")
        if st.button("Submit_Log_Parag"):
            st.text("Using NLP ..")
            cl = cleaning(message_log_parag)
            vec = vectorizer_parag.transform(pd.Series(cl))
            result = model_log_parag.predict_proba(vec)
            st.success('The probility for Frau Merkel is {pro:.2f}'.format(pro=result[0][0]))
            st.success('The probility for Frau Steinmeir is {pro:.2f}'.format(pro=result[0][1]))


    if st.checkbox("Show Logistic Analysis"):
        with gzip.open('model_log.pklz', 'rb') as ifp:
            model_log = pickle.load(ifp)
        st.subheader("Text Analysis")
        message_log = st.text_area("Enter Your Text    ")
        if st.button("Submit_Log"):
            st.text("Using NLP ..")
            cl=cleaning(message_log)
            vec = vector.transform(pd.Series(cl))
            result=model_log.predict_proba(vec)
            st.success('The probility for Frau Merkel is {pro:.2f}'.format(pro=result[0][0]))    
            st.success('The probility for Frau Steinmeir is {pro:.2f}'.format(pro=result[0][1]))
            
    if st.checkbox("Show RandomForest Analysis"):
        with gzip.open('model_random.pklz', 'rb') as ifp:
            model_rf1 = pickle.load(ifp)
        st.subheader("Text Analysis")
        message_rf1 = st.text_area("Your Text    ")
        if st.button("Submit_Random"):
            st.text("Using NLP ..")
            cl=cleaning(message_rf1)
            vec = vector.transform(pd.Series(cl))
            result=model_rf1.predict_proba(vec)
            st.success('The probility for Frau Merkel is {pro:.2f}'.format(pro=result[0][0]))    
            st.success('The probility for Frau Steinmeir is {pro:.2f}'.format(pro=result[0][1]))
            
    if st.checkbox("Show XGB Analysis"):
        with gzip.open('xgb.pklz', 'rb') as ifp:
            model_xgb = pickle.load(ifp)
        st.subheader("Text Analysis")
        message_xgb = st.text_area("Text    ")
        if st.button("Submit_XBG"):
            st.text("Using NLP ..")
            cl=cleaning(message_xgb)
            vec = vector.transform(pd.Series(cl))
            result=model_xgb.predict_proba(vec)
            st.success('The probility for Frau Merkel is {pro:.2f}'.format(pro=result[0][0]))    
            st.success('The probility for Frau Steinmeir is {pro:.2f}'.format(pro=result[0][1]))
            

    if st.checkbox("Word Calculation in the contect of two leaders"):
        st.subheader("Text Analysis")
        first = st.text_area("Enter first word")
        pos = st.text_area("Enter a word to extract from first word")
        neg = st.text_area("Enter a word to add to first word")
        if st.button("Submit_Equation"):
            st.text("Using NLP ..")
            result = vectorizer_emding.most_similar(positive=[first, pos], negative=[neg], topn=1)
            st.success('If we are extracting {} from {}, and then adding {} , we will have {}'.format(neg, first, pos, result[0][0]))

    if st.checkbox("Word Context of two leaders"):
        st.subheader("Text Analysis")
        first = st.text_area("Enter word")
        if st.button("Submit_Word"):
            st.text("Using NLP ..")
            result = vectorizer_emding.most_similar(first)
            st.success('The most three similar words to {} are: {}, {}, {}'.format(first, result[0][0], result[1][0], result[2][0]))
    
st.sidebar.subheader("About App")
st.sidebar.info("NLP App with Streamlit.")
                
    
    
if __name__ == '__main__':
    main()

