from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk import download
import string

download('punkt')
download('averaged_perceptron_tagger')
download('wordnet')
download('omw-1.4')
download('stopwords')


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
def remove_punctuation(tokens):
    return list(filter(lambda token: token not in string.punctuation, tokens))

def lemmatize_sentence(sentence):
    return [lemmatizer.lemmatize(token,tag) if tag in ['J','V','N','R'] else lemmatizer.lemmatize(token) for (token,tag) in [(token,get_wordnet_pos(tag)) for (token,tag) in sentence]]
    
def remove_stopwords(sentence):
        return [w.lower() for w in sentence if not w.lower() in stop_words]

def preprocess_text(text):
    try:
        return remove_stopwords(lemmatize_sentence(pos_tag(remove_punctuation(word_tokenize(text)))))
    except:
        print(text)

def subreddit_keywords(df,subreddit_id):
    popular_in_subreddit=df[df['subreddit_id']==subreddit_id]['preprocessed_body'].explode().value_counts().reset_index().head(150)["index"].to_list()
    popular_in_other=df[df['subreddit_id']!=subreddit_id]['preprocessed_body'].explode().value_counts().reset_index().head(150)["index"].to_list()
    return [keyword for keyword in popular_in_subreddit if keyword not in popular_in_other]