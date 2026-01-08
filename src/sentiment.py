'''
This program looks at the varying sentiments of the top 5 names from each ethnic group. 
'''

import numpy as np
import pandas as pd
import pickle
import re
from time import time

from tqdm import tqdm
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


def words_to_sentiment(words, embeddings, model):
    """
    Given a list of words, the embeddings matrix, and the trained model,
    return the sentiment scores for each word as a pandas DataFrame.
    If there are any words that are not in the embeddings matrix, this
    returns an empty DataFrame.
    """
    found_words = [word for word in words if word in embeddings.index]
    if len(found_words) < len(words):
        print("Warning: some words not found. Returning empty dataframe.")
        return pd.DataFrame()
    vecs = embeddings.loc[found_words]
    predictions = model.predict_log_proba(vecs)
    log_odds = predictions[:, 1] - predictions[:, 0]
    return pd.DataFrame({'sentiment': log_odds}, index=vecs.index)

def text_to_sentiment(text, embeddings, model):
    """
    Given a string, the embeddings matrix, and the trained model,
    tokenize the string, compute the sentiment for each token, and
    return the average sentiment score. If there are any words
    that are not in the embeddings matrix, this returns 0.0.
    """
    tokenize_rgx = re.compile(r"\w.*?\b")
    tokens = [token.lower() for token in tokenize_rgx.findall(text)]
    sentiments = words_to_sentiment(tokens, embeddings, model)
    if len(sentiments) == 0:
        return 0.0
    return sentiments['sentiment'].mean()


def main():
    datadir = Path('/data/glove/')
    embeddings = pickle.load(open(datadir / 'glove.42B.300d.pkl', 'rb'))

    lexdir = Path('/data/opinion-lexicon-English/')
    positive = open(lexdir/'positive-words.txt').read().splitlines()
    negative = open(lexdir/'negative-words.txt').read().splitlines()
    positive = [word for word in positive if word in embeddings.index]
    negative = [word for word in negative if word in embeddings.index]

    ethinic_dict = {
        "white": ["yoder", "friedman", "krueger", "schwartz", "schmitt"],
        "black": ["washington", "jefferson", "mosley", "charles", "jackson"],
        "asian": ["xiong", "zhang", "huang", "truong", "huynh"],
        "indian": ["sampson", "jacobs", "lucero", "ashley", "cummings"],
        "multi": ["ali", "wong", "singh", "ahmed", "chung"],
        "hispanic": ["barajas", "zavala", "velazquez", "avalos", "vazquez"]
    }

    slabels = positive + negative
    svectors = embeddings.loc[slabels]  
    stargets = np.array([1]*len(positive) + [-1]*len(negative))

    train_vecs, test_vecs, train_tgts, test_tgts, train_labels, test_labels = \
        train_test_split(svectors, stargets, slabels, test_size=0.1, random_state=0)
    model = SGDClassifier(loss='log_loss', random_state=0, max_iter=100)
    model.fit(train_vecs, train_tgts)

    for ethincity, names in ethinic_dict.items():
        print(ethincity)
        for name in names:
            print(f"Sentiment for '{name}': {text_to_sentiment(name, embeddings, model)}")
        print()

main()