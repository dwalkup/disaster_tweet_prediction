

import re, nltk, string
from bs4 import BeautifulSoup

def text_cleaner(text):
    '''Cleans tweet text for modeling.
    
    Changes text to lower case, removes hyperlinks, @users, html tags,
    punctuation, words with numbers inside them, and numbers.
    
    Args:
        text (string): the text to be cleaned
        
    Returns:
        text (string): the cleaned text
    '''
    
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    soup = BeautifulSoup(text, 'lxml')
    text = soup.get_text()
    text = re.sub('%20', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\d', '', text)
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    text = tokenizer.tokenize(text)
    text = ' '.join(text)
    return text

def text_normalize(text, stem_it = False, lemmatize_it = False):
    '''Normalizes tweet text for modeling.
    
    Tokenizes text, stems or lemmatizes the text if desired, then rejoins
    the text back into a single string.
    
    Args:
        text (string): the text to be normalized
        stem_it (boolean): whether to stem the input text.
            Default = False
        lemmatize_it (boolean): whether to lemmatize the input text
            Default = False
        
    Returns:
        text (string): the normalized text
    '''
    
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    stemmer = nltk.PorterStemmer()
    lemmatizer = nltk.WordNetLemmatizer()
    text = tokenizer.tokenize(text)
    if stem_it:
        text = [stemmer.stem(word) for word in text]
    elif lemmatize_it:
        text = [lemmatizer.lemmatize(word) for word in text]
    else:
        return text
    text = ' '.join(text)
    return text

