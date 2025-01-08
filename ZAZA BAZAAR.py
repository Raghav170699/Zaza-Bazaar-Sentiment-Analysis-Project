#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[4]:


df = pd.rad_csv('Dataset.csv')
df.head()


# In[5]:


nltk.download('vader_lexicon')


# In[14]:


sid = SentimentIntensityAnalyzer()


# In[20]:


def get_sentiment_scores(series):
    """
    Returns a list of dictionaries, where each dictionary includes the compound, positive,
    negative, and neutral scores for a given text in the input series.
    """
    sentiment_scores = []
    for text in series:
        sentiment_scores.append(sid.polarity_scores(str(text)))
    return sentiment_scores

# Calculate the sentiment scores for the partial_entry column
sentiment_scores = get_sentiment_scores(df['partial_entry'])

# Print the sentiment scores
print(sentiment_scores)


# In[25]:


import pandas as pd
import nltk
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')

# Load the dataset
df = pd.read_csv('Dataset.csv')

# Instantiate the SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Define a function to get the sentiment score for each entry in a Series
def get_sentiment_scores(series):
    """
    Returns a list of dictionaries, where each dictionary includes the compound, positive,
    negative, and neutral scores for a given text in the input series.
    """
    sentiment_scores = []
    for text in series:
        sentiment_scores.append(sid.polarity_scores(str(text)))
    return sentiment_scores

# Calculate the sentiment scores for the partial_entry column
sentiment_scores = pd.Series(get_sentiment_scores(df['partial_entry']))

# Extract the counts of each sentiment
counts = sentiment_scores.apply(lambda x: 'positive' if x['compound'] > 0.05 else ('negative' if x['compound'] < -0.05 else 'neutral')).value_counts()

# Create a bar chart of the counts
fig, ax = plt.subplots()
ax.bar(counts.index, counts.values)
ax.set_title('Sentiment Analysis')
ax.set_xlabel('Sentiment')
ax.set_ylabel('Count')
plt.show()


# In[26]:


import pandas as pd
import nltk
nltk.download('vader_lexicon')
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Dataset.csv')

# Instantiate the SentimentIntensityAnalyzer
sid = nltk.sentiment.vader.SentimentIntensityAnalyzer()

# Define a function to get the sentiment score for each entry in a Series
def get_sentiment_scores(series):
    """
    Returns a list of dictionaries, where each dictionary includes the compound, positive,
    negative, and neutral scores for a given text in the input series.
    """
    sentiment_scores = []
    for text in series:
        sentiment_scores.append(sid.polarity_scores(str(text)))
    return sentiment_scores

# Calculate the sentiment scores for the partial_entry column
sentiment_scores = get_sentiment_scores(df['partial_entry'])

# Count the number of positive, negative, and neutral reviews
num_pos = sum(score['compound'] > 0 for score in sentiment_scores)
num_neg = sum(score['compound'] < 0 for score in sentiment_scores)
num_neu = sum(score['compound'] == 0 for score in sentiment_scores)

# Create a bar chart of the counts
fig, ax = plt.subplots()
ax.bar(['Positive', 'Negative', 'Neutral'], [num_pos, num_neg, num_neu], color=['green', 'red', 'blue'])
ax.set_title('Sentiment Analysis')
ax.set_xlabel('Sentiment')
ax.set_ylabel('Count')
plt.show()


# In[45]:


pip install wordcloud


# In[46]:


nltk.download('punkt')


# In[47]:


nltk.download('stopwords')


# In[74]:


pip install pyLDAvis


# In[75]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import gensim
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Dataset.csv')

# Instantiate the SentimentIntensityAnalyzer
sid = nltk.sentiment.vader.SentimentIntensityAnalyzer()

# Define a function to preprocess the text data
def preprocess_text(text):
    # Tokenize the text and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]

    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

# Define a function to get the sentiment score for each entry in a Series
def get_sentiment_scores(series):
    """
    Returns a list of dictionaries, where each dictionary includes the compound, positive,
    negative, and neutral scores for a given text in the input series.
    """
    sentiment_scores = []
    for text in series:
        sentiment_scores.append(sid.polarity_scores(str(text)))
    return sentiment_scores

# Calculate the sentiment scores for the partial_entry column
sentiment_scores = get_sentiment_scores(df['partial_entry'])

# Add the sentiment scores to the DataFrame
df = pd.concat([df, pd.DataFrame(sentiment_scores)], axis=1)

# Filter out neutral reviews
df = df[df['compound'] != 0]

# Preprocess the text data
df['tokens'] = df['partial_entry'].apply(preprocess_text)

# Create a dictionary and corpus for the LDA model
dictionary = corpora.Dictionary(df['tokens'])
corpus = [dictionary.doc2bow(tokens) for tokens in df['tokens']]

# Train the LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=dictionary,
                                            num_topics=5,
                                            random_state=42,
                                            passes=10)

# Print the topics and their top words
for topic_num, topic_words in lda_model.show_topics(num_topics=5, num_words=10):
    print(f'Topic #{topic_num+1}: {topic_words}')

# Visualize the topics
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis)


# In[76]:


nltk.download('wordnet')


# In[77]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import gensim
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Dataset.csv')

# Instantiate the SentimentIntensityAnalyzer
sid = nltk.sentiment.vader.SentimentIntensityAnalyzer()

# Define a function to preprocess the text data
def preprocess_text(text):
    # Tokenize the text and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]

    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

# Define a function to get the sentiment score for each entry in a Series
def get_sentiment_scores(series):
    """
    Returns a list of dictionaries, where each dictionary includes the compound, positive,
    negative, and neutral scores for a given text in the input series.
    """
    sentiment_scores = []
    for text in series:
        sentiment_scores.append(sid.polarity_scores(str(text)))
    return sentiment_scores

# Calculate the sentiment scores for the partial_entry column
sentiment_scores = get_sentiment_scores(df['partial_entry'])

# Add the sentiment scores to the DataFrame
df = pd.concat([df, pd.DataFrame(sentiment_scores)], axis=1)

# Filter out neutral reviews
df = df[df['compound'] != 0]

# Preprocess the text data
df['tokens'] = df['partial_entry'].apply(preprocess_text)

# Create a dictionary and corpus for the LDA model
dictionary = corpora.Dictionary(df['tokens'])
corpus = [dictionary.doc2bow(tokens) for tokens in df['tokens']]

# Train the LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=dictionary,
                                            num_topics=5,
                                            random_state=42,
                                            passes=10)

# Print the topics and their top words
for topic_num, topic_words in lda_model.show_topics(num_topics=5, num_words=10):
    print(f'Topic #{topic_num+1}: {topic_words}')

# Visualize the topics
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis)


# In[78]:


nltk.download('omw-1.4')


# In[80]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import gensim
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Dataset.csv')

# Instantiate the SentimentIntensityAnalyzer
sid = nltk.sentiment.vader.SentimentIntensityAnalyzer()

# Define a function to preprocess the text data
def preprocess_text(text):
    # Tokenize the text and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]

    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

# Define a function to get the sentiment score for each entry in a Series
def get_sentiment_scores(series):
    """
    Returns a list of dictionaries, where each dictionary includes the compound, positive,
    negative, and neutral scores for a given text in the input series.
    """
    sentiment_scores = []
    for text in series:
        sentiment_scores.append(sid.polarity_scores(str(text)))
    return sentiment_scores

# Calculate the sentiment scores for the partial_entry column
sentiment_scores = get_sentiment_scores(df['partial_entry'])

# Add the sentiment scores to the DataFrame
df = pd.concat([df, pd.DataFrame(sentiment_scores)], axis=1)

# Filter out neutral reviews
df = df[df['compound'] != 0]

# Preprocess the text data
df['tokens'] = df['partial_entry'].apply(preprocess_text)

# Create a dictionary and corpus for the LDA model
dictionary = corpora.Dictionary(df['tokens'])
corpus = [dictionary.doc2bow(tokens) for tokens in df['tokens']]

# Train the LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=dictionary,
                                            num_topics=5,
                                            random_state=42,
                                            passes=10)

# Print the topics and their top words
for topic_num, topic_words in lda_model.show_topics(num_topics=5, num_words=10):
    print(f'Topic #{topic_num+1}: {topic_words}')

# Visualize the topics
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis)


# In[98]:


import matplotlib.pyplot as plt
import numpy as np

# Define the topics and their corresponding word weights
topics = {
    'Topic #1': {'food': 0.051, 'time': 0.026, 'eat': 0.019, 'get': 0.018, 'like': 0.018, 'choice': 0.018, 'place': 0.016, 'good': 0.016, 'load': 0.015, 'za': 0.014},
    'Topic #2': {'food': 0.062, 'family': 0.034, 'experience': 0.027, 'good': 0.027, 'ate': 0.023, 'member': 0.021, 'everyone': 0.018, 'nice': 0.018, 'variety': 0.018, 'time': 0.017},
    'Topic #3': {'food': 0.075, 'fun': 0.045, 'love': 0.042, 'also': 0.038, 'kid': 0.03, 'quality': 0.023, 'variety': 0.022, 'choice': 0.021, 'great': 0.015, 'place': 0.014},
    'Topic #4': {'lunch': 0.07, 'full': 0.042, 'came': 0.027, 'section': 0.026, 'choice': 0.025, 'everyone': 0.023, 'work': 0.023, 'many': 0.022, 'limited': 0.021, 'expecting': 0.02},
    'Topic #5': {'night': 0.042, 'seat': 0.031, 'good': 0.03, 'really': 0.022, 'go': 0.022, 'window': 0.02, 'table': 0.019, 'went': 0.019, 'drink': 0.017, 'bar': 0.014}
}

# Create a bar graph for each topic
for i, topic in enumerate(topics.keys()):
    # Sort the words by weight in descending order
    sorted_words = sorted(topics[topic], key=topics[topic].get, reverse=True)
    weights = [topics[topic][word] for word in sorted_words]
    
    # Set up the plot
    plt.figure(i+1, figsize=(10, 6))
    plt.bar(np.arange(len(sorted_words)), weights, align='center', alpha=0.5)
    plt.xticks(np.arange(len(sorted_words)), sorted_words)
    plt.ylabel('Word Weight')
    plt.title(topic)
    
    # Display the plot
    plt.show()


# In[97]:


import pandas as pd
import nltk
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Dataset.csv')

# Instantiate the SentimentIntensityAnalyzer
sid = nltk.sentiment.vader.SentimentIntensityAnalyzer()

# Define a function to get the sentiment score for each entry in a Series
def get_sentiment_scores(series):
    """
    Returns a list of lists, where each inner list includes the compound, positive,
    negative, and neutral scores for a given text in the input series.
    """
    sentiment_scores = []
    for text in series:
        score = list(sid.polarity_scores(str(text)).values())
        sentiment_scores.append(score)
    return sentiment_scores

# Calculate the sentiment scores for the partial_entry column
sentiment_scores = get_sentiment_scores(df['partial_entry'])

# Perform k-means clustering
kmeans = KMeans(n_clusters=3).fit(sentiment_scores)
labels = kmeans.labels_

# Plot the clusters
plt.scatter([score[1] for score in sentiment_scores], [score[2] for score in sentiment_scores], 
            s=[score[0]*100 for score in sentiment_scores], c=labels)
plt.xlabel('Positive Sentiment Score')
plt.ylabel('Negative Sentiment Score')
plt.title('Sentiment Clustering')
plt.show()


# In[88]:


import pandas as pd
import nltk
nltk.download('vader_lexicon')
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Dataset.csv')

# Instantiate the SentimentIntensityAnalyzer
sid = nltk.sentiment.vader.SentimentIntensityAnalyzer()

# Define a function to get the sentiment score for each entry in a Series
def get_sentiment_scores(series):
    """
    Returns a list of dictionaries, where each dictionary includes the compound, positive,
    negative, and neutral scores for a given text in the input series.
    """
    sentiment_scores = []
    for text in series:
        sentiment_scores.append(sid.polarity_scores(str(text)))
    return sentiment_scores

# Calculate the sentiment scores for the partial_entry column
sentiment_scores = get_sentiment_scores(df['partial_entry'])

# Create a new dataframe with the sentiment scores as columns
sentiment_df = pd.DataFrame(sentiment_scores)

# Fit a K-means model on the sentiment scores
kmeans = KMeans(n_clusters=5, random_state=0).fit(sentiment_df)

# Create a DataFrame with the data and the cluster assignments
df_clusters = pd.DataFrame({'data': df['partial_entry'], 'cluster': kmeans.labels_})

# Group the data by cluster
cluster_groups = df_clusters.groupby('cluster')
    
def print_top_terms_per_cluster(df, vectorizer, kmeans, num_terms=10):
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()

# Print the top terms per cluster
for cluster, group in cluster_groups:
    print(f'Cluster {cluster}:')
    top_words = group['data'].str.split(expand=True).stack().value_counts().head(10)
    print(top_words)
    print('\n')


# In[89]:


# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Create a list to hold the negative words
negative_words = []

# Loop through each entry in the 'partial_entry' column and extract negative words
for text in df['partial_entry'].astype(str):
    # Get the sentiment scores for the text
    scores = sid.polarity_scores(text)
    # If the negative score is greater than the positive score, consider it a negative text
    if scores['neg'] > scores['pos']:
        # Split the text into words
        words = text.split()
        # Loop through each word and check if it has a negative score
        for word in words:
            if sid.polarity_scores(word)['neg'] > 0:
                negative_words.append(word)

# Remove duplicates from the negative words list
negative_words = list(set(negative_words))

# Print the negative words
print(negative_words)


# In[92]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the stop words corpus
nltk.download('stopwords')

# Load dataframe
df = pd.read_csv('your_combined_dataset.csv')

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Define stop words
stop_words = set(stopwords.words('english'))

# Calculate sentiment scores for each entry in the 'partial_entry' column and remove stop words
sentiment_scores = []
for text in df['partial_entry'].astype(str):
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    text_clean = ' '.join(words)
    sentiment_scores.append(sid.polarity_scores(text_clean))

# Create a new dataframe with the sentiment scores
df_sentiment_scores = pd.DataFrame(sentiment_scores)

# Combine the original dataframe with the sentiment scores dataframe
df_combined = pd.concat([df, df_sentiment_scores], axis=1)

# Print the top 10 most negative words
words = []
for text in df['partial_entry'].astype(str):
    words += nltk.word_tokenize(text.lower())
words = [word for word in words if word.isalpha() and word not in stop_words]
neg_words = []
for word in words:
    score = sid.polarity_scores(word)['compound']
    if score < 0:
        neg_words.append(word)
print(pd.Series(neg_words).value_counts().nlargest(10))


# In[96]:


import pandas as pd
import nltk
nltk.download('vader_lexicon')
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load the dataset
df = pd.read_csv('Dataset.csv')

# Instantiate the SentimentIntensityAnalyzer
sid = nltk.sentiment.vader.SentimentIntensityAnalyzer()

# Define a function to get the sentiment score for each entry in a Series
def get_sentiment_scores(series):
    """
    Returns a list of dictionaries, where each dictionary includes the compound, positive,
    negative, and neutral scores for a given text in the input series.
    """
    sentiment_scores = []
    for text in series:
        sentiment_scores.append(sid.polarity_scores(str(text)))
    return sentiment_scores

# Calculate the sentiment scores for the partial_entry column
sentiment_scores = get_sentiment_scores(df['partial_entry'])

# Create a new dataframe with the sentiment scores as columns
sentiment_df = pd.DataFrame(sentiment_scores)

# Fit a K-means model on the sentiment scores
kmeans = KMeans(n_clusters=5, random_state=0).fit(sentiment_df)

# Create a DataFrame with the data and the cluster assignments
df_clusters = pd.DataFrame({'data': df['partial_entry'], 'cluster': kmeans.labels_})

# Combine all text of each cluster into a single string
cluster_text = df_clusters.groupby('cluster')['data'].apply(lambda x: ' '.join(x))

# Create a CountVectorizer object to extract features
vectorizer = CountVectorizer(stop_words='english')

# Fit the vectorizer on the text of each cluster
X = vectorizer.fit_transform(cluster_text)

# Get the feature names
terms = vectorizer.get_feature_names()

# Create a list to hold the top terms for each cluster
top_terms = []

# Get the top terms for each cluster and add them to the list
for i in range(5):
    cluster_terms = []
    for ind in X[i].toarray().argsort()[0][-10:]:
        cluster_terms.append(terms[ind])
    top_terms.append(cluster_terms)

# Print the top terms per cluster
print("Top terms per cluster:")
for i in range(5):
    print(f"Cluster {i}: {', '.join(top_terms[i])}")


# In[99]:


import matplotlib.pyplot as plt
import numpy as np

# Define the topics and their corresponding word weights
topics = {
    'Topic #1': {'food': 0.051, 'time': 0.026, 'eat': 0.019, 'get': 0.018, 'like': 0.018, 'choice': 0.018, 'place': 0.016, 'good': 0.016, 'load': 0.015, 'za': 0.014},
    'Topic #2': {'food': 0.062, 'family': 0.034, 'experience': 0.027, 'good': 0.027, 'ate': 0.023, 'member': 0.021, 'everyone': 0.018, 'nice': 0.018, 'variety': 0.018, 'time': 0.017},
    'Topic #3': {'food': 0.075, 'fun': 0.045, 'love': 0.042, 'also': 0.038, 'kid': 0.03, 'quality': 0.023, 'variety': 0.022, 'choice': 0.021, 'great': 0.015, 'place': 0.014},
    'Topic #4': {'lunch': 0.07, 'full': 0.042, 'came': 0.027, 'section': 0.026, 'choice': 0.025, 'everyone': 0.023, 'work': 0.023, 'many': 0.022, 'limited': 0.021, 'expecting': 0.02},
    'Topic #5': {'night': 0.042, 'seat': 0.031, 'good': 0.03, 'really': 0.022, 'go': 0.022, 'window': 0.02, 'table': 0.019, 'went': 0.019, 'drink': 0.017, 'bar': 0.014}
}

# Define a color map for the bars
cmap = plt.get_cmap('tab10')

# Create a bar graph for each topic
for i, topic in enumerate(topics.keys()):
    # Sort the words by weight in descending order
    sorted_words = sorted(topics[topic], key=topics[topic].get, reverse=True)
    weights = [topics[topic][word] for word in sorted_words]
    
    # Set up the plot
    plt.figure(i+1, figsize=(10, 6))
    for j, word in enumerate(sorted_words):
        color = cmap(j/len(sorted_words))
        plt.bar(j, weights[j], align='center', alpha=0.5, color=color)
    plt.xticks(np.arange(len(sorted_words)), sorted_words)
    plt.ylabel('Word Weight')
    plt.title(topic)
    
    # Display the plot
    plt.show()


# In[100]:


import matplotlib.pyplot as plt
import numpy as np

# Define the topics and their corresponding word weights
topics = {
    'Topic #1': {'food': 0.051, 'time': 0.026, 'eat': 0.019, 'get': 0.018, 'like': 0.018, 'choice': 0.018, 'place': 0.016, 'good': 0.016, 'load': 0.015, 'za': 0.014},
    'Topic #2': {'food': 0.062, 'family': 0.034, 'experience': 0.027, 'good': 0.027, 'ate': 0.023, 'member': 0.021, 'everyone': 0.018, 'nice': 0.018, 'variety': 0.018, 'time': 0.017},
    'Topic #3': {'food': 0.075, 'fun': 0.045, 'love': 0.042, 'also': 0.038, 'kid': 0.03, 'quality': 0.023, 'variety': 0.022, 'choice': 0.021, 'great': 0.015, 'place': 0.014},
    'Topic #4': {'lunch': 0.07, 'full': 0.042, 'came': 0.027, 'section': 0.026, 'choice': 0.025, 'everyone': 0.023, 'work': 0.023, 'many': 0.022, 'limited': 0.021, 'expecting': 0.02},
    'Topic #5': {'night': 0.042, 'seat': 0.031, 'good': 0.03, 'really': 0.022, 'go': 0.022, 'window': 0.02, 'table': 0.019, 'went': 0.019, 'drink': 0.017, 'bar': 0.014}
}

# Define the colors to be used for the bars in each graph
colors = ['r', 'g', 'b', 'c', 'm']

# Create a bar graph for each topic
for i, topic in enumerate(topics.keys()):
    # Sort the words by weight in descending order
    sorted_words = sorted(topics[topic], key=topics[topic].get, reverse=True)
    weights = [topics[topic][word] for word in sorted_words]
    
    # Set up the plot
    plt.figure(i+1, figsize=(10, 6))
    plt.bar(np.arange(len(sorted_words)), weights, align='center', alpha=0.5, color=colors[i%len(colors)])
    plt.xticks(np.arange(len(sorted_words)), sorted_words)
    plt.ylabel('Word Weight')
    plt.title(topic)
    
    # Display the plot
    plt.show()


# In[ ]:




