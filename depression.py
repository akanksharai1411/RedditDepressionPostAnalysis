import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt


nltk.download('punkt')
nltk.download('stopwords')

#Read in the data
df = pd.read_csv('depression-sampled.csv')

df = df.drop(df.columns[0], axis=1)

#Total number of posts
print("Total number of posts: ", len(df))

#Total number of unique authors
print("Total number of unique authors: ", len((df['author'].unique())))


#Average post length (measured in word count)
df['word_count'] = df['selftext'].apply(lambda x: len(str(x).split(" ")))
print("Average post length (measured in word count): ", df['word_count'].mean())

#Date range of the post
df= df[pd.to_numeric(df['created_utc'], errors='coerce').notnull()]
df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
print("Date range of the dataset: ", df['created_utc'].min(), " - ", df['created_utc'].max())

#Top 20 most important words in the post

#Remove empty spaces
df['selftext'] = df['selftext'].fillna('')

# Remove punctuation
df['selftext'] = df['selftext'].str.replace('[^\w\s]','')

# Remove stopwords (the, and, a)
stop = stopwords.words('english')  #create a list
df['selftext'] = df['selftext'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# Remove numbers
df['selftext'] = df['selftext'].str.replace('\d+', '')

# Remove short words
df['selftext'] = df['selftext'].apply(lambda x: " ".join(x for x in x.split() if len(x) > 3))

# Convert to lowercase
df['selftext'] = df['selftext'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Tokenize (returns a list of lists with all punctuation and whitespace removed)
df['selftext'] = df['selftext'].apply(word_tokenize)

# Lemmatize (sort them and group together variant groups)
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
df['selftext'] = df['selftext'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Join the list of words back into a string
df['selftext'] = df['selftext'].apply(lambda x: ' '.join(x))

# Count the frequency of each word
freq = pd.Series(' '.join(df['selftext']).split()).value_counts()[:20]
print("Top 20 most important words in the posts which are related to depression: ", freq)

#remove commonly used words
freq = list(freq.index)
df['selftext'] = df['selftext'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))



#Visualise the most common words in the post using wordcloud 

#checks for floats
df['selftext'] = df['selftext'].apply(lambda x: '' if isinstance(x, float) else x)

# Generate a string of all the selftext values in the dataframe
text = " ".join(df['selftext'])

# Create the wordcloud object
wordcloud = WordCloud(width=800, height=800, background_color='white', max_words=200, contour_width=3, contour_color='steelblue')

# Generate the wordcloud
wordcloud.generate(text)

# Visualize the wordcloud
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

# Show the plot
plt.show()






