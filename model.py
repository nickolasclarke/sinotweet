#%%
import re
import pickle

import numpy as np
import pandas as pd
import gensim
import jieba


#%% 
# read in data from pickle, extract only the tweets
tweets = pd.read_pickle('data/tweets.pickle')
tweets = tweets['tweet_text']

# %%
#begin cleaning tweets, extract only Chinese characters
stop_words = np.loadtxt('data/cn_stopwords.txt',dtype=str)
#def clean_data(text):
#    cleaned_text = re.sub(r"[\s\/\\_$^*(+\"\'+~\-@#&^*:\[\]{}【】]+", "", str(text))
#    return cleaned_text
def getChinese(context):
    '''TODO
    '''
    #context = context.decode("utf-8") # convert context from str to unicode
    filtrate = re.compile(u'[^\u4E00-\u9FA5]') # non-Chinese unicode range
    context = filtrate.sub(r'', context) # remove all non-Chinese characters
    #context = context.encode("utf-8") # convert unicode back to str
    return context
# %%
def cut_remove_stopwords(context):
    '''TODO
    '''
    context = jieba.lcut(context)
    cleaned = [word for word in context if word not in stop_words]
    #cleaned = ", ".join([word for word in context if word not in stop_words])
    return cleaned

# %%
tweets = pd.Series([getChinese(tweet) for tweet in tweets])
tweets = tweets.replace('', np.nan)
tweets = tweets[tweets.isna() == False]
tweets = pd.Series([cut_remove_stopwords(tweet) for tweet in tweets])


# %%
# Prepare gensim

def get_lda(corpus, words_no_above=0.5, num_topics=10):
    '''TODO
    Create an LDA model
    '''
    dictionary = gensim.corpora.Dictionary(corpus)
    dictionary.filter_extremes(no_above=words_no_above)
    bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=num_topics, id2word=dictionary)
    topic_dic = dict()
    for idx, topic in lda_model.print_topics(-1):
        topic_dic[f'topic {idx}'] = topic.split('+')
    return pd.DataFrame(topic_dic)
# %%
#TODO run against a subset to optimize.
get_lda(tweets.sample(10000), words_no_above=0.6, num_topics=5)
# %%
