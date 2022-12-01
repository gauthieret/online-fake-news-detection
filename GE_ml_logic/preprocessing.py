from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import unidecode


def preprocessing(sentence):

    # Basic cleaning
    sentence = sentence.strip() ## remove whitespaces
    sentence = sentence.lower() ## lowercase
    sentence = ''.join(char for char in sentence if not char.isdigit()) ## remove numbers

    # Advanced cleaning
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '') ## remove punctuation

    unaccented_string = unidecode.unidecode(sentence) # remove accents

    tokenized_sentence = word_tokenize(unaccented_string) ## tokenize
    stop_words = set(stopwords.words('english')) ## define stopwords

    tokenized_sentence_cleaned = [ ## remove stopwords
    w for w in tokenized_sentence if not w in stop_words
            ]

    lemmatized = [
    WordNetLemmatizer().lemmatize(word, pos = "v")
    for word in tokenized_sentence_cleaned
    ]

    cleaned_sentence = ' '.join(word for word in lemmatized)


    return cleaned_sentence
