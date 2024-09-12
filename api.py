# -*- coding: utf-8 -*-
"""API for Text Similarity"""

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import requests
from bs4 import BeautifulSoup
import re

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

nlp = spacy.load('en_core_web_sm')

url = 'https://www.bbc.com/news'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')


scraped_titles = [h2.get_text() for h2 in soup.select('[data-testid="card-text-wrapper"] h2') if h2.get_text()]
scraped_docs = [p.get_text() for p in soup.select('[data-testid="card-text-wrapper"] p') if p.get_text()]

def generate_keywords(text):
    doc = nlp(text)
    keywords = set()
    
    # Extract named entities and noun chunks as keywords
    for entity in doc.ents:
        keywords.add(entity.text)
    for chunk in doc.noun_chunks:
        keywords.add(chunk.text)
    
    return list(keywords)

scraped_keywords = set()
for doc in scraped_docs:
    keywords = generate_keywords(doc)
    scraped_keywords.update(keywords)

scraped_keywords = list(scraped_keywords)

stopwords_list = stopwords.words('english')

# Existing variables and preprocessing
english_stopset = set(stopwords.words('english')).union(
                  {"things", "that's", "something", "take", "don't", "may", "want", "you're",
                   "set", "might", "says", "including", "lot", "much", "said", "know",
                   "good", "step", "often", "going", "thing", "things", "think",
                   "back", "actually", "better", "look", "find", "right", "example",
                                                                  "verb", "verbs"})

docs = scraped_docs
title = scraped_titles
keywords = scraped_keywords

# Preprocessing
documents_clean = []
documents_cleant = []

for d in docs:
    document_test = re.sub(r'[^\x00-\x7F]+', ' ', d)  #Replace non-ASCII characters with space
    document_test = re.sub(r'@\w+', '', document_test)  #eliminate duplicate whitespaces/ # Remove Mentions
    document_test = document_test.lower() #converting to lower
    document_test = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document_test) #cleaning punctuation
    document_test = re.sub(r'[0-9]', '', document_test) #replacing number with empity string
    document_test = re.sub(r'\s{2,}', ' ', document_test)
    documents_clean.append(document_test)
    documents_cleant.append(document_test)

lemmatizer = WordNetLemmatizer()

new_doc = [' '.join([lemmatizer.lemmatize(docs) for docs in text.split(',')]) for text in docs]
new_title = [' '.join([lemmatizer.lemmatize(title).strip() for title in text.split(' ')])for text in title]

english_stopset = list(stopwords.words('english'))

vectorizer = TfidfVectorizer(analyzer='word',
                             ngram_range=(1,2),
                             min_df=0.002,
                             max_df=0.99,
                             max_features=10000,
                             lowercase=True,
                             stop_words=english_stopset)

X = vectorizer.fit_transform(new_doc)
df = pd.DataFrame(X.T.toarray())

# API route
@app.route('/similarity', methods=['POST'])
@cross_origin()
def get_similar_articles_api():
    data = request.json
    q = data.get('query')
    t = data.get('title')
    top_results = int(data.get('top_results', 10))  # Default to 10 if not provided
    
    if not q or not t:
        return jsonify({"error": "Please provide both 'query' and 'title' in the request."}), 400
    
    q = [q]
    t = [t]

    q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0],)
    q_vect = vectorizer.transform(t).toarray().reshape(df.shape[0],)
    
    sim = {}
    titl = {}
    
    for i in range(len(new_doc)) and range(len(new_title)):
        sim[i] = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)
        titl[i] = np.dot(df.loc[:, i].values, q_vect) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vect)

    sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)[:min(len(sim), top_results)]
    sim_sortedt = sorted(titl.items(), key=lambda x: x[1], reverse=True)[:min(len(titl), top_results)]

    results = []
    for (i, v), (j, vt) in zip(sim_sorted, sim_sortedt):
        if v != 0.0:
            results.append({
                "similarity_score": v,
                "title_similarity_score": vt,
                "title": new_title[i],
                "document": new_doc[i]
            })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
