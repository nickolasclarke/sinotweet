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
tweets_2019 = pd.read_pickle('data/tweets2019.pickle')

tweets = tweets['tweet_text']
tweets_2019 = tweets_2019['tweet_text']
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
def clean_tweets(df):
    df = pd.Series([getChinese(tweet) for tweet in df])
    df = df.replace('', np.nan)
    df = df[df.isna() == False]
    df = pd.Series([cut_remove_stopwords(tweet) for tweet in df])
    return df

# %%


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
#%%
tweets = clean_tweets(tweets)
tweets_2019 = clean_tweets(tweets_2019)
# %%
#TODO run against a subset to optimize.
tweets_model = get_lda(tweets.sample(80000), words_no_above=0.9, num_topics=3)
tweets_model
# %%
tweets2019_model = get_lda(tweets_2019.sample(100000), words_no_above=0.5, num_topics=5)
tweets2019_model

# %%
