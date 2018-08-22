import numpy as np 
import pandas as pd 
import nltk
from nltk.corpus import stopwords
import time
import spacy
from scipy.misc import imread
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import pickle

start_time = time.time()
train_df = pd.read_csv("train.csv", encoding='latin-1')
test_df = pd.read_csv("test.csv", encoding='latin-1')
end = time.time()
print("Time taken in reading the input files is {}.".format(end - start_time))
train_df.head()


# Explore the dataset 
print("Number of rows in train dataset {} ".format(train_df.shape[0]))

import string
def unique_word_fraction(row):
    """function to calculate the fraction of unique words on total words of the text"""
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    unique_count = list(set(text_splited)).__len__()
    return (unique_count/float(word_count))

def words_count(row):
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    return len(text_splited)

def char_count(row):
    """function to return number of chracters """
    return len(row['text'])

def frac_alpha_char(row):
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    no_alpha_char = 0
    for w in text_splited:
        no_alpha_char += len([i for i in w if i.isalpha()])
    return (no_alpha_char/float(word_count))

eng_stopwords = set(stopwords.words("english"))
def stopwords_count(row):
    """ Number of stopwords fraction in a text"""
    text = row['text'].lower()
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    stopwords_count = len([w for w in text_splited if w in eng_stopwords])
    return (stopwords_count/float(word_count))

def punctuations_fraction(row):
    """functiopn to claculate the fraction of punctuations over total number of characters for a given text """
    text = row['text']
    char_count = len(text)
    punctuation_count = len([c for c in text if c in string.punctuation])
    return (punctuation_count/float(char_count))

def quotes_usage(row):
    """functiopn to claculate the fraction of punctuations over total number of characters for a given text """
    text = row['text']
    quote_count = len([c for c in text if c == "\""])
    return quote_count

def quotes_single_usage(row):
    """functiopn to claculate the fraction of punctuations over total number of characters for a given text """
    text = row['text']
    quote_count = len([c for c in text if c == "\'"])
    return quote_count

def comma_usage(row):
    """functiopn to claculate the fraction of punctuations over total number of characters for a given text """
    text = row['text']
    comma_count = len([c for c in text if c == ","])
    return comma_count

def fraction_noun(row):
    """function to give us fraction of noun over total words """
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    noun_count = len([w for w in pos_list if w[1] in ('NN','NNP','NNPS','NNS')])
    return (noun_count/float(word_count))

def fraction_pronoun(row):
    """function to give us fraction of noun over total words """
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    pronoun_count = len([w for w in pos_list if w[1] in ('PRP','PRP$','WP','WP$')])
    return (pronoun_count/float(word_count))

def male_pronoun(row):
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    male_pronoun_count = 0
    for w in pos_list:
        if w[1] in ('PRP','PRP$','WP','WP$'):
            if w[0].lower() in ('he','him','his','himself'):
                male_pronoun_count+=1
    return male_pronoun_count

def female_pronoun(row):
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    female_pronoun_count = 0
    for w in pos_list:
        if w[1] in ('PRP','PRP$','WP','WP$'):
            if w[0].lower() in ('she','her','hers','herself'):
                female_pronoun_count+=1
    return female_pronoun_count

def fraction_adj(row):
    """function to give us fraction of adjectives over total words in given text"""
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    adj_count = len([w for w in pos_list if w[1] in ('JJ','JJR','JJS')])
    return (adj_count/float(word_count))

def fraction_verbs(row):
    """function to give us fraction of verbs over total words in given text"""
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    verbs_count = len([w for w in pos_list if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])
    return (verbs_count/float(word_count))

def num_word_upper_Frac(row):
    """function to give us fraction of verbs over total words in given text"""
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    num_word_upper_Frac = len([w for w in text_splited if w.isupper()])
    return (num_word_upper_Frac/float(word_count))

def num_word_title_Frac(row):
    """function to give us fraction of verbs over total words in given text"""
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    num_word_upper_Frac = len([w for w in text_splited if w.istitle()])
    return (num_word_upper_Frac/float(word_count))

def num_names(row):
    """function to give us fraction of verbs over total words in given text"""
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    name_count = len([w for w in pos_list if w[1] in ('NNP')])
    return (name_count)

def num_names_Frac(row):
    """function to give us fraction of verbs over total words in given text"""
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    sentt = nltk.ne_chunk(pos_list, binary = False)
    name_count = len([w for w in pos_list if w[1] in ('NNP')])
    return (name_count/float(word_count))

def first_name(row):
    """function to give us fraction of verbs over total words in given text"""
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    first_name = 0
    if pos_list[0][1] == 'NNP':
        first_name = 1
    return (first_name)

def avg_word_len(row):
    """function to give us fraction of average length of words in given text"""
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    sum_word_lengths = (''.join(text_splited)).__len__()
    return (sum_word_lengths/float(word_count))

def max_word_len(row):
    """function to give us fraction of average length of words in given text"""
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    max_word_len = 0
    for w in text_splited:
        if len(w) > max_word_len:
            max_word_len = len(w)
    return max_word_len

#avg_WordLen_dict = {'EAP':4.466905, 'HPL':4.514069, 'MWS':4.443011}

def fracWordsLenGreaterThanAvg(row):
    """function to give us fraction of verbs over total words in given text"""
    text = row['text']
    avg_word_len = row['avg_word_len']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    fracWordsLenGreaterThanAvg = len([w for w in text_splited if len(w) > avg_word_len])
    return fracWordsLenGreaterThanAvg

from textblob import TextBlob  
def sentiment(row):
    """function to give us fraction of verbs over total words in given text"""
    text = row['text']
    analysis = TextBlob(text)
    sentiment = 0
    if (analysis.sentiment.polarity > 0):
        sentiment = 1
    if (analysis.sentiment.polarity < 0):
        sentiment = -1
    return (sentiment)

def TF_IDF_Afterwards_Ind(row):
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    TF_IDF_Afterwards_Ind = 0
    for w in text_splited:
        if w[0].lower() == "afterwards":
            TF_IDF_Afterwards_Ind = 1
    return TF_IDF_Afterwards_Ind

def TF_IDF_Thanks_Ind(row):
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    TF_IDF_Thanks_Ind = 0
    for w in text_splited:
        if w[0].lower() == "thanks":
            TF_IDF_Thanks_Ind = 1
    return TF_IDF_Thanks_Ind

def TF_IDF_Younger_Ind(row):
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    TF_IDF_Younger_Ind = 0
    for w in text_splited:
        if w[0].lower() == "younger":
            TF_IDF_Younger_Ind = 1
    return TF_IDF_Younger_Ind

def TF_IDF_Later_Ind(row):
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    TF_IDF_Later_Ind = 0
    for w in text_splited:
        if w[0].lower() == "later":
            TF_IDF_Later_Ind = 1
    return TF_IDF_Later_Ind

def TF_IDF_Amongst_Ind(row):
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    TF_IDF_Amongst_Ind = 0
    for w in text_splited:
        if w[0].lower() == "amongst":
            TF_IDF_Amongst_Ind = 1
    return TF_IDF_Amongst_Ind

def TF_IDF_Dont_Ind(row):
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    TF_IDF_Dont_Ind = 0
    for w in text_splited:
        if w[0].lower() == "don\'t":
            TF_IDF_Dont_Ind = 1
    return TF_IDF_Dont_Ind

def freq_char(row, x):
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    freq_char = 0
    for w in text_splited:
        freq_char += len([i for i in w if i.lower() == x])
    return (freq_char/float(word_count))

print("Style based features train_df")
train_df['unique_word_fraction'] = train_df.apply(lambda row: unique_word_fraction(row), axis =1)
train_df['words_count'] = train_df.apply(lambda row: words_count(row), axis =1)
train_df['char_count'] = train_df.apply(lambda row: char_count(row), axis =1)
train_df['frac_alpha_char'] = train_df.apply(lambda row: frac_alpha_char(row), axis =1)
train_df['stopwords_count'] = train_df.apply(lambda row: stopwords_count(row), axis =1)
train_df['punctuations_fraction'] = train_df.apply(lambda row: punctuations_fraction(row), axis =1)
train_df['quote_count'] = train_df.apply(lambda row: quotes_usage(row), axis =1)
train_df['comma_count'] = train_df.apply(lambda row: comma_usage(row), axis =1)
train_df['quotes_single_usage'] = train_df.apply(lambda row: quotes_single_usage(row), axis =1)
train_df['fraction_noun'] = train_df.apply(lambda row: fraction_noun(row), axis =1)
train_df['fraction_pronoun'] = train_df.apply(lambda row: fraction_pronoun(row), axis =1)
train_df['male_pronoun_count'] = train_df.apply(lambda row: male_pronoun(row), axis =1)
train_df['female_pronoun_count'] = train_df.apply(lambda row: female_pronoun(row), axis =1)
train_df['fraction_adj'] = train_df.apply(lambda row: fraction_adj(row), axis =1)
train_df['fraction_verbs'] = train_df.apply(lambda row: fraction_verbs(row), axis =1)
train_df['num_word_upper_Frac'] = train_df.apply(lambda row: num_word_upper_Frac(row), axis =1)
train_df['num_word_title_Frac'] = train_df.apply(lambda row: num_word_title_Frac(row), axis =1)
train_df['num_names'] = train_df.apply(lambda row: num_names(row), axis =1)
train_df['num_names_Frac'] = train_df.apply(lambda row: num_names_Frac(row), axis =1)
train_df['is_first_name'] = train_df.apply(lambda row: first_name(row), axis =1)
train_df['avg_word_len'] = train_df.apply(lambda row: avg_word_len(row), axis =1)
train_df['fracWordsLenGreaterThanAvg'] = train_df.apply(lambda row: fracWordsLenGreaterThanAvg(row), axis =1)
train_df['max_word_len'] = train_df.apply(lambda row: max_word_len(row), axis =1)
train_df['sentiment'] = train_df.apply(lambda row: sentiment(row), axis =1)
train_df['frac_freq_a'] = train_df.apply(lambda row: freq_char(row,"a"), axis =1)
train_df['frac_freq_b'] = train_df.apply(lambda row: freq_char(row,"b"), axis =1)
train_df['frac_freq_c'] = train_df.apply(lambda row: freq_char(row,"c"), axis =1)
train_df['frac_freq_d'] = train_df.apply(lambda row: freq_char(row,"d"), axis =1)
train_df['frac_freq_e'] = train_df.apply(lambda row: freq_char(row,"e"), axis =1)
train_df['frac_freq_f'] = train_df.apply(lambda row: freq_char(row,"f"), axis =1)
train_df['frac_freq_g'] = train_df.apply(lambda row: freq_char(row,"g"), axis =1)
train_df['frac_freq_h'] = train_df.apply(lambda row: freq_char(row,"h"), axis =1)
train_df['frac_freq_i'] = train_df.apply(lambda row: freq_char(row,"i"), axis =1)
train_df['frac_freq_j'] = train_df.apply(lambda row: freq_char(row,"j"), axis =1)
train_df['frac_freq_k'] = train_df.apply(lambda row: freq_char(row,"k"), axis =1)
train_df['frac_freq_l'] = train_df.apply(lambda row: freq_char(row,"l"), axis =1)
train_df['frac_freq_m'] = train_df.apply(lambda row: freq_char(row,"m"), axis =1)
train_df['frac_freq_n'] = train_df.apply(lambda row: freq_char(row,"n"), axis =1)
train_df['frac_freq_o'] = train_df.apply(lambda row: freq_char(row,"o"), axis =1)
train_df['frac_freq_p'] = train_df.apply(lambda row: freq_char(row,"p"), axis =1)
train_df['frac_freq_q'] = train_df.apply(lambda row: freq_char(row,"q"), axis =1)
train_df['frac_freq_r'] = train_df.apply(lambda row: freq_char(row,"r"), axis =1)
train_df['frac_freq_s'] = train_df.apply(lambda row: freq_char(row,"s"), axis =1)
train_df['frac_freq_t'] = train_df.apply(lambda row: freq_char(row,"t"), axis =1)
train_df['frac_freq_u'] = train_df.apply(lambda row: freq_char(row,"u"), axis =1)
train_df['frac_freq_v'] = train_df.apply(lambda row: freq_char(row,"v"), axis =1)
train_df['frac_freq_w'] = train_df.apply(lambda row: freq_char(row,"w"), axis =1)
train_df['frac_freq_x'] = train_df.apply(lambda row: freq_char(row,"x"), axis =1)
train_df['frac_freq_y'] = train_df.apply(lambda row: freq_char(row,"y"), axis =1)
train_df['frac_freq_z'] = train_df.apply(lambda row: freq_char(row,"z"), axis =1)

train_df['TF_IDF_Afterwards_Ind'] = train_df.apply(lambda row: TF_IDF_Afterwards_Ind(row), axis =1)
train_df['TF_IDF_Younger_Ind'] = train_df.apply(lambda row: TF_IDF_Younger_Ind(row), axis =1)
train_df['TF_IDF_Thanks_Ind'] = train_df.apply(lambda row: TF_IDF_Thanks_Ind(row), axis =1)
train_df['TF_IDF_Later_Ind'] = train_df.apply(lambda row: TF_IDF_Later_Ind(row), axis =1)
train_df['TF_IDF_Amongst_Ind'] = train_df.apply(lambda row: TF_IDF_Amongst_Ind(row), axis =1)
train_df['TF_IDF_Dont_Ind'] = train_df.apply(lambda row: TF_IDF_Dont_Ind(row), axis =1)

train_df.head()

print("Style based features test_df")
#test_df['unique_word_fraction'] = test_df.apply(lambda row: unique_word_fraction(row), axis =1)
#test_df['words_count'] = test_df.apply(lambda row: words_count(row), axis =1)
#test_df['char_count'] = test_df.apply(lambda row: char_count(row), axis =1)
#test_df['stopwords_count'] = test_df.apply(lambda row: stopwords_count(row), axis =1)
#test_df['punctuations_fraction'] = test_df.apply(lambda row: punctuations_fraction(row), axis =1)
#test_df['fraction_noun'] = test_df.apply(lambda row: fraction_noun(row), axis =1)
#test_df['fraction_pronoun'] = test_df.apply(lambda row: fraction_pronoun(row), axis =1)
#test_df['fraction_adj'] = test_df.apply(lambda row: fraction_adj(row), axis =1)
#test_df['fraction_verbs'] = test_df.apply(lambda row: fraction_verbs(row), axis =1)
#test_df['num_word_upper_Frac'] = test_df.apply(lambda row: num_word_upper_Frac(row), axis =1)
#test_df['num_word_title_Frac'] = test_df.apply(lambda row: num_word_title_Frac(row), axis =1)
#test_df['num_names'] = test_df.apply(lambda row: num_names(row), axis =1)
#test_df['num_names_Frac'] = test_df.apply(lambda row: num_names_Frac(row), axis =1)
#test_df['is_first_name'] = test_df.apply(lambda row: first_name(row), axis =1)
#test_df['avg_word_len'] = test_df.apply(lambda row: avg_word_len(row), axis =1)
#test_df['fracWordsLenGreaterThanAvg'] = test_df.apply(lambda row: fracWordsLenGreaterThanAvg(row), axis =1)
#test_df['sentiment'] = test_df.apply(lambda row: sentiment(row), axis =1)
#test_df.head()
print("Style based features complete")

start = time.time()
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes
tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3))                #<class 'sklearn.feature_extraction.text.TfidfVectorizer'>
full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist() + test_df['text'].values.tolist())    #<class 'scipy.sparse.csr.csr_matrix'>
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())                 #<class 'scipy.sparse.csr.csr_matrix'>
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())                   #<class 'scipy.sparse.csr.csr_matrix'>
end = time.time()
print("Time taken in tf-idf is {}.".format(end-start))

cols_to_drop = ['id', 'text']
train_X = train_df.drop(cols_to_drop+['author'], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)
author_mapping_dict = {'EAP':0, 'HPL':1, 'MWS':2}
train_y = train_df['author'].map(author_mapping_dict)

#multinomial Naive Bayes
def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])                                       #(19579, 3)
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5.

#SVD on word TFIDF:
n_comp = 20
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
    
train_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]
test_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)
del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd

# Naive Bayes on Word Count Vectorizer:
# Fit transform the count vectorizer ###
tfidf_vec = CountVectorizer(stop_words='english', ngram_range=(1,3))
tfidf_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5.

# add the predictions as new features #
train_df["NaiveBayes_Word_EAP"] = pred_train[:,0]
train_df["NaiveBayes_Word_HPL"] = pred_train[:,1]
train_df["NaiveBayes_Word_MWS"] = pred_train[:,2]
test_df["NaiveBayes_Word_EAP"] = pred_full_test[:,0]
test_df["NaiveBayes_Word_HPL"] = pred_full_test[:,1]
test_df["NaiveBayes_Word_MWS"] = pred_full_test[:,2]

#Naive Bayes on Character Count Vectorizer:
### Fit transform the tfidf vectorizer ###
tfidf_vec = CountVectorizer(ngram_range=(1,7), analyzer='char')
tfidf_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5.

# add the predictions as new features #
train_df["nb_cvec_char_eap"] = pred_train[:,0]
train_df["nb_cvec_char_hpl"] = pred_train[:,1]
train_df["nb_cvec_char_mws"] = pred_train[:,2]
test_df["nb_cvec_char_eap"] = pred_full_test[:,0]
test_df["nb_cvec_char_hpl"] = pred_full_test[:,1]
test_df["nb_cvec_char_mws"] = pred_full_test[:,2]

#Naive Bayes on Character Tfidf Vectorizer:
### Fit transform the tfidf vectorizer ###
tfidf_vec = TfidfVectorizer(ngram_range=(1,5), analyzer='char')
full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist() + test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5.

# add the predictions as new features #
train_df["nb_tfidf_char_eap"] = pred_train[:,0]
train_df["nb_tfidf_char_hpl"] = pred_train[:,1]
train_df["nb_tfidf_char_mws"] = pred_train[:,2]
test_df["nb_tfidf_char_eap"] = pred_full_test[:,0]
test_df["nb_tfidf_char_hpl"] = pred_full_test[:,1]
test_df["nb_tfidf_char_mws"] = pred_full_test[:,2]

#SVD on Character TFIDF:
n_comp = 20
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
    
train_svd.columns = ['svd_char_'+str(i) for i in range(n_comp)]
test_svd.columns = ['svd_char_'+str(i) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)
del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd

# split train into trn and val for cross validation
val_df = train_df.sample(frac=0.2, replace=False)
trn_df = train_df.drop(val_df.index)
print("trn_df.shape[0]",trn_df.shape[0])
print("val_df.shape[0]",val_df.shape[0])
print("train_df.shape[0]",train_df.shape[0])
trn_df.head()

import random

class Dictogram(dict):
    def __init__(self, iterable=None):
        """Initialize this histogram as a new dict; update with given items"""
        super(Dictogram, self).__init__()
        self.types = 0  # the number of distinct item types in this histogram
        self.tokens = 0  # the total count of all item tokens in this histogram
        if iterable:
            self.update(iterable)
    def update(self, iterable):
        """Update this histogram with the items in the given iterable"""
        for item in iterable:
            if item in self:
                self[item] += 1
                self.tokens += 1
            else:
                self[item] = 1
                self.types += 1
                self.tokens += 1
    def count(self, item):
        """Return the count of the given item in this histogram, or 0"""
        if item in self:
            return self[item]
        return 0
    def return_random_word(self):
        # Another way:  Should test: random.choice(histogram.keys())
        random_key = random.sample(self, 1)
        return random_key[0]
    def return_weighted_random_word(self):
        # Step 1: Generate random number between 0 and total count - 1
        random_int = random.randint(0, self.tokens-1)
        index = 0
        list_of_keys = self.keys()
        # print 'the random index is:', random_int
        for i in range(0, self.types):
            index += self[list_of_keys[i]]
            # print index
            if(index > random_int):
                # print list_of_keys[i]
                return list_of_keys[i]

# markov chain based features, 3 words memory 
def make_higher_order_markov_model(order, data):
    markov_model = dict()
    for i in range(0, len(data)-order):
        # Create the window
        window = tuple(data[i: i+order])
        # Add to the dictionary
        if window in markov_model:
            # We have to just append to the existing Dictogram
            markov_model[window].update([data[i+order]])
        else:
            markov_model[window] = Dictogram([data[i+order]])
    return markov_model

print(trn_df.loc[train_df['author']=='EAP'].shape[0])
def tokenixed_list(row):
    """function to calculate the fraction of unique words on total words of the text"""
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    return (text_splited)

trn_df['splited_text'] = trn_df.apply(lambda row: tokenixed_list(row), axis =1)

eap = trn_df.loc[trn_df['author']=='EAP']['splited_text'].values
flat_eap = []
eap_flat = [flat_eap.extend(x) for x in eap]
flat_eap.__len__()
eap_MM = make_higher_order_markov_model(3, flat_eap)

hpl = trn_df.loc[trn_df['author']=='HPL']['splited_text'].values
flat_hpl = []
hpl_flat = [flat_hpl.extend(x) for x in hpl]
flat_hpl.__len__()
hpl_MM = make_higher_order_markov_model(3, flat_hpl)

ms = trn_df.loc[trn_df['author']=='MWS']['splited_text'].values
flat_ms = []
ms_flat = [flat_ms.extend(x) for x in ms]
flat_ms.__len__()
ms_MM = make_higher_order_markov_model(3, flat_ms)

#probablity calculation using markov chain
#print(eap_MM[('no', 'means', 'of')])
#print(sum([x for x in eap_MM[('no', 'means', 'of')].values()]))
def spit_out_prob(tupl , word , mm1= eap_MM, mm2 = hpl_MM, mm3 = ms_MM):
    """function to spit out a prob of a word given a tuple and a word and a markov model"""
    """fix this function, its working but not calculating probabilities, but providing
    some confidance index, greater than confidance index is, larger is the prob"""
    try:
        a = mm1[tupl][word]
    except:
        a = 0 
    try:
        b = mm2[tupl][word]
    except:
        b = 0
    try:
        c = mm3[tupl][word]
    except:
        c = 0
    try:
        prob_word = a/(a+b+c)#(sum([x for x in mm[tupl].values()]))
        return (prob_word)
    except:
        return (0) #earlier it was 1



def make_tupl(sentence =trn_df.loc[train_df['author']=='EAP']['splited_text'][0] , n=3):
    """function to make tuples of n size given a sentence and n"""
    list_of_tuple = []
    word_1 = []
    for i in list(range(sentence.__len__()-3)):
        tuple_1 = (sentence[i], sentence[i+1], sentence[i+2])
        list_of_tuple.append(tuple_1)
        word_1.append(sentence[i+3])
    return (list_of_tuple, word_1)


def sent_to_prob_eap(sentence, mm1 = eap_MM, p_eap =1/3, mm2 = hpl_MM, mm3 = ms_MM):
    """function to get the markov model to give prob of a author given a sentence """
    list_of_tuples = make_tupl(sentence, n =3)[0]
    words = make_tupl(sentence, n =3)[1]
    #print(list_of_tuples, words)
    #p = p_eap 
    p=0
    for i in list(range(words.__len__())):
        #print(list_of_tuples[i], words[i])
        p = p+spit_out_prob(list_of_tuples[i], words[i], mm1 =eap_MM, mm2 =hpl_MM, mm3 =ms_MM)
    return(p)

def sent_to_prob_hpl(sentence, mm2 = hpl_MM, p_hpl =1/3, mm1 = eap_MM, mm3 = ms_MM):
    """function to get the markov model to give prob of a author given a sentence """
    list_of_tuples = make_tupl(sentence, n =3)[0]
    words = make_tupl(sentence, n =3)[1]
    #print(list_of_tuples, words)
    #p = p_hpl 
    p = 0
    for i in list(range(words.__len__())):
        #print(list_of_tuples[i], words[i])
        p = p+spit_out_prob(list_of_tuples[i], words[i], mm1 =hpl_MM, mm2=eap_MM, mm3=ms_MM)
    return(p)

def sent_to_prob_ms(sentence, mm3 = ms_MM, p_ms =1/3, mm2 = hpl_MM, mm1 = eap_MM):
    """function to get the markov model to give prob of a author given a sentence """
    list_of_tuples = make_tupl(sentence, n =3)[0]
    words = make_tupl(sentence, n =3)[1]
    #print(list_of_tuples, words)
    #p = p_ms 
    p = 0
    for i in list(range(words.__len__())):
        #print(list_of_tuples[i], words[i])
        p = p+spit_out_prob(list_of_tuples[i], words[i], ms_MM, hpl_MM, eap_MM)
    return(p)

# calculating markov chain based features
trn_df['EAP_markov_3'] = trn_df['splited_text'].apply(lambda row: sent_to_prob_eap(sentence =row))
trn_df['HPL_markov_3'] = trn_df['splited_text'].apply(lambda row: sent_to_prob_hpl(sentence = row))
trn_df['MWS_markov_3'] = trn_df['splited_text'].apply(lambda row: sent_to_prob_ms(sentence = row))
trn_df[['MWS_markov_3', 'EAP_markov_3', 'HPL_markov_3', 'author']].head()
del trn_df['splited_text']

val_df['splited_text'] = val_df.apply(lambda row: tokenixed_list(row), axis =1)
val_df['EAP_markov_3'] = val_df['splited_text'].apply(lambda row: sent_to_prob_eap(sentence =row))
val_df['HPL_markov_3'] = val_df['splited_text'].apply(lambda row: sent_to_prob_hpl(sentence = row))
val_df['MWS_markov_3'] = val_df['splited_text'].apply(lambda row: sent_to_prob_ms(sentence = row))
val_df1 = val_df[['MWS_markov_3', 'EAP_markov_3', 'HPL_markov_3','author']].copy()
del val_df['splited_text']

test_df['splited_text'] = test_df.apply(lambda row: tokenixed_list(row), axis =1)
test_df['EAP_markov_3'] = test_df['splited_text'].apply(lambda row: sent_to_prob_eap(sentence =row))
test_df['HPL_markov_3'] = test_df['splited_text'].apply(lambda row: sent_to_prob_hpl(sentence = row))
test_df['MWS_markov_3'] = test_df['splited_text'].apply(lambda row: sent_to_prob_ms(sentence = row))
test_df1 = test_df[['MWS_markov_3', 'EAP_markov_3', 'HPL_markov_3']].copy()
del test_df['splited_text']

print ("trn_df.head()")
print trn_df.head()
print ("trn_df.shape and column names after Markov Features",trn_df.shape)
print list(trn_df.columns.values)

author_mapping_dict = {'EAP':0, 'HPL':1, 'MWS':2}
trn_y = trn_df['author'].map(author_mapping_dict)
val_y = val_df['author'].map(author_mapping_dict)
cols_drop = ['id', 'text','MWS_markov_3', 'EAP_markov_3', 'HPL_markov_3']
trn_X = trn_df.drop(cols_drop+['author'], axis=1)
val_X = val_df.drop(cols_drop+['author'], axis=1)
#test_X = test_df.drop(cols_drop, axis=1)

trn_X.to_pickle("trn_X.pickle")
val_X.to_pickle("val_X.pickle")
trn_y.to_pickle("trn_y.pickle")
val_y.to_pickle("val_y.pickle")
val_df1.to_pickle("val_df1.pickle")