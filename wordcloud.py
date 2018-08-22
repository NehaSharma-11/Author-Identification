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
import wordcloud as WC


start_time = time.time()
train_df = pd.read_csv("train.csv", encoding='latin-1')
test_df = pd.read_csv("test.csv", encoding='latin-1')
end = time.time()
print("Time taken in reading the input files is {}.".format(end - start_time))
train_df.head()

#from wordcloud import WordCloud, STOPWORDS
eap = train_df[train_df.author=="EAP"]["text"].values
hpl = train_df[train_df.author=="HPL"]["text"].values
mws = train_df[train_df.author=="MWS"]["text"].values

wc = WC.WordCloud(background_color="black", max_words=5000, 
               stopwords=STOPWORDS, max_font_size= 50)
# generate word cloud
wc.generate(" ".join(train_df.text.values))

# show
plt.figure(figsize=(16,13))
plt.imshow(wc, interpolation='bilinear')
plt.title("words from all author", fontsize=14,color='seagreen')
plt.axis("off")