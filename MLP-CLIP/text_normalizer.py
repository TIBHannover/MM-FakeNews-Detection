import re, string, unicodedata

def replace_contractions(text):
    """Replace contractions in string of text"""
    return text

def remove_URL(sample):
    """Remove URLs from a sample string"""
    sample = re.sub(r"\S+\.[(net)|(com)|(org)]\S+", "", sample)
    sample = re.sub(r"http\S+", "", sample)
    sample = re.sub(r"\d+", " ", sample)
    sample = re.sub(r"\s+", " ", sample)
    sample = re.sub(r"_", " ", sample)
    return sample

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', ' ', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    return words

def preprocess(sample):
    sample = remove_URL(sample)
    # Tokenize
    words = sample.split(' ')
    words = normalize(words)

    normalized_text = ''
    for w in words:
        normalized_text += w+' '

    return normalized_text.strip()