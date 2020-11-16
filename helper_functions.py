import praw
import contractions
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd
import gensim
from gensim.models import CoherenceModel
import gensim.corpora as corpora
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tag import pos_tag
from nltk.tokenize import WhitespaceTokenizer, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



def get_title_link(label):
    '''
    Use praw to scrape titles of posts and to get the permalink for comments.
    Store these in a list and then make a dataframe.
    Args: The keyword to search in the Futurology subreddit.
    Returns: A dataframe consisting the titles and permalinks of the posts consisting of the keyword.
    '''
    reddit = praw.Reddit(client_id='1zU0VluRtDiEUg',
                     client_secret='SUuSJCLllaOIDWM_gkX_Bn86gQ1uBw',
                     user_agent='Reddit_scrape_meat')
    sub = reddit.subreddit('Futurology')
    titles = []
    permalink = []
    for submission in reddit.subreddit("Futurology").search(label, limit = 10000):
        titles.append(submission.title)
        permalink.append(submission.permalink)
    df1 = pd.DataFrame(titles)
    df2 = pd.DataFrame(permalink)
    df = pd.concat([df1,df2], axis = 1)
    df.columns = ['title','permalink']
    return df

def get_comments(df):
    '''
    Use the titles and permalinks scraped using the get_title_link function.
    Get the comments and time posted related to the titles.
    Index corresponds to the title dataframe index in order to match the comment with title.
    Args: The dataframe consisting of titles and permalinks.
    Returns: A dataframe consisting the comments, time created, and indexes.
    '''
    reddit = praw.Reddit(client_id='1zU0VluRtDiEUg',
                     client_secret='SUuSJCLllaOIDWM_gkX_Bn86gQ1uBw',
                     user_agent='Reddit_scrape_meat')
    comment_list = []
    index = []
    time_list = []
    for idx, i in enumerate(df['permalink']):
        url = "https://www.reddit.com" + i
        submission = reddit.submission(url = url)
        submission.comments.replace_more(limit=None, threshold=0)
        for comment in submission.comments.list():
            comment_list.append(comment.body)
            index.append(idx)
            time_list.append(comment.created_utc)
    df1 = pd.DataFrame(index)
    df2 = pd.DataFrame(comment_list)
    df3 = pd.DataFrame(time_list)
    df = pd.concat([df1,df2,df3],axis=1)
    df.columns = ['index','comment','time']
    return df


def text_cleaning(df, column):
    '''
    Clean the comment dataframe.
    Make all comments a string, then expand contractions, remove punctuations and stop words.
    Finally, lemmatize the words with WordNetLemmatizer. A new column is created for each step
    Args: A dataframe to clean.
    Returns: A dataframe consisting the titles and permalinks of the posts.
    '''
    stop_words = set(stopwords.words('english'))

    #add words that aren't in the NLTK stopwords list
    new_stopwords = ['lol', 'yup', 'nope' ,'http', 'www', 'com', 'https','http','deleted','removed','would','I','oh','org','people','animal','']
    new_stopwords_list = stop_words.union(new_stopwords)
    def remove_stopwords(text):
        return " ".join([word for word in str(text).split() if word not in new_stopwords_list])
    def expand_word(text):
        return " ".join([contractions.fix(word) for word in str(text).split()])
    def lemmatize_text(text):
        w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
        lemmatizer = nltk.stem.WordNetLemmatizer()
        return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
    df[column] = df[column].str.lower()
    df['expand'] = df[column].apply(lambda x: expand_word(x))
    df['no_punct'] = df['expand'].str.replace('[^\w\s]',' ')
    df['no_stop'] = df['no_punct'].apply(lambda x: remove_stopwords(x))
    df['lemm'] = df['no_stop'].apply(lambda x: lemmatize_text(x))

    return df


def prepare_for_LDA(df, column):
    '''
    Prepare data for LDA modeling. Create id2word, texts and a corpus to input into the LDA model.
    Args: The dataframe consisting of the info to prepare for modeling.
    Returns: A dataframe consisting the id2word, texts and corpus as LDA model inputs.
    '''
    columns = df[column]
    column_dict = [x.split() for x in columns]
    id2word = corpora.Dictionary(column_dict)
    texts = column_dict
    corpus = [id2word.doc2bow(text) for text in texts]

    return id2word, texts, corpus

def get_topics_coherence(model, corpus, texts, dictionary):
    '''
    Check the coherence score after running the LDA model to check if the topic keywords make sense
    when put together.
    Args: The model, corpus, texts, dictionary.
    Returns: A model's topic coherence.
    '''
    pprint(model.print_topics())
    doc = model[corpus]
    coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence = coherence_model.get_coherence()
    print('\nCoherence Score: ', coherence)


def get_topic_df (df, model, corpus, texts):
    """
    Borrowed code from https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
    Get the topics for the model and create a dataframe with dominant topic, percent contribution, and topic keywords appended.
    Args: The dataframe, model, corpus and texts
    Returns: New columns appended to the dataframe consisting of the dominant topic of each document, the percent contribution of that dominant topic in the text,
    and the topic keywards.
    """
    topics_df = pd.DataFrame()
    for i, row_list in enumerate(model[corpus]):
        row = row_list[0] if model.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j ==0:
                wp = model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                topics_df = topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    topics_df.columns = ['Dominant_Topic', 'Percent_Contribution', 'Topic_keywords']
    contents = pd.Series(texts)
    topics_df = pd.concat([topics_df, df], axis = 1)
    return topics_df




def get_sentiment_distribution(topics_df, label):
    '''
    To get the results of the topic dataframe and model for analysis.
    Add a couple more stopwords in so the wordclouds could make more sense.
    Use vader sentiment analysis to check the average sentiment of each topic.
    The compound score is used, which looks at the overall positive and negative sentiment
    and provides a combined score.
    Finally, a word cloud is created to visualize the top keywords in each topic.
    The distribution of the sentiment was also plotted in a graph.
    Args: The dataframe consisting of the topics and a keyword to name the new wordcloud files.
    Returns: A a sentiment score, sentiment distribution for that topic. A wordcloud for that topic. 
    '''
    stop_words = set(stopwords.words('english'))
    new_stopwords = ['gov', 'pastebin','fcrfs94k','nlm','ncbi','thank','I','make','lol', 'yup', 'nope' ,'http', 'www', 'com', 'https','http','deleted','removed','think','would','like','meat','know']
    new_stopwords_list = stop_words.union(new_stopwords)
    for i in range(int(max(topics_df['Dominant_Topic'])+1)):
        sid = SentimentIntensityAnalyzer()
        topic = topics_df['Dominant_Topic'] == i
        topic_df = topics_df[topic]
        topic_df['vader'] = topics_df['comment'].apply(lambda comment:sid.polarity_scores(comment))
        topic_df['compound'] = topic_df['vader'].apply(lambda score_dict: score_dict['compound'])
        print('The sentiment score for this topic is {}.'.format(round(topic_df['compound'].mean(),4)))
        wordcloud = WordCloud(background_color = "GhostWhite", stopwords = new_stopwords_list).generate(' '.join(topics_df['comment']))
        wordcloud.to_image()
        wordcloud.to_file('wordcloud_'+label+str(i)+'.png')

        num_bins = 50
        plt.figure(figsize=(10,6))
        n, bins, patches = plt.hist(topic_df.compound, num_bins, facecolor='blue', alpha=0.5)
        plt.xlabel('Polarity')
        plt.ylabel('Count')
        plt.title('Histogram of polarity of comments for Topic'+str(i))
        plt.show()
        print('-------------------------------------------------')
