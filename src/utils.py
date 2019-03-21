import os
import re
import requests
import conllu

from nltk.corpus import stopwords

from gensim import downloader
from gensim.utils import simple_preprocess

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
print(BASE_DIR)

def get_conll_dataset(lang='en'):
    """
    Get dataset from github repo: https://github.com/ufal/rh_nntagging
    """
    files = []
    repo = 'https://raw.githubusercontent.com/ufal/rh_nntagging/master/data/ud-1.2/'
    urls = {"train": os.path.join(repo, lang, lang + '-ud-train.conllu'),
            "dev": os.path.join(repo, lang, lang + '-ud-dev.conllu'),
            "test": os.path.join(repo, lang, lang + '-ud-test.conllu'), }

    # create language folder if not exist
    if not os.path.exists(os.path.join(BASE_DIR, 'data', lang)):
        os.makedirs(os.path.join(BASE_DIR, 'data', lang))

    for filename, url in urls.items():
        filepath = os.path.join(BASE_DIR, 'data', lang, filename + '.conllu')
        files.append((filename, filepath))

        # check if already downloaded
        if os.path.isfile(filepath):
            continue

        r = requests.get(url, allow_redirects=True)

        with open(filepath, 'wb') as output:
            output.write(r.content)

    return files


def read_text_dataset(filename):
    """
    This handle the free text file, not in conllu format
    """
    with open(filename) as f:
        data = f.read()
        sentences = re.split(r' *[\.\?!][\'"\)\]]* *', data)

        tokenized_sentences = []

        for s in sentences:
            # clean text of websites, email address and any punctuation
            text = re.sub(r"((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", s)
            tokens = simple_preprocess(text)
            tokenized_sentences.append(tokens)

    return tokenized_sentences


def get_gensim_dataset(name):
    corpus = downloader.load(name)
    # gensim return a interable object which is list of tokenized corpus

    return corpus


def parse_conll(files):
    """
    Because the dataset in conllu format, so we need to parse it for processing
    """
    data = {}
    for file in files:
        with open(file[1], "r", encoding="utf-8") as f:
            sentences = list(tokens for tokens in conllu.parse_incr(f))
        data[file[0]] = sentences

    return data


def clean_data(data, lang='english'):
    sentences = []
    stop_words = set(stopwords.words(lang))

    for sentence in data:
        text = []
        for token in sentence:
            # token object is OrderDict type with lemma as one key
            # It looks like this
            # OrderedDict([('id', 1), ('form', 'What'), ('lemma', 'what'), ('upostag', 'PRON')])
            text.append(token['lemma'])
        # merge list to string
        text = ' '.join(text)
        # clean text of websites, email address and any punctuation
        text = re.sub(r"((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)
        # tokenize with gensim
        tokens = simple_preprocess(text)
        # clean stop words
        tokens = [w for w in tokens if not w in stop_words]
        sentences.append(tokens)

    return sentences


def get_data(lang="english"):
    # download dataset from server
    files = get_conll_dataset(lang=lang[:2])

    # parse the conllu format to python object
    # return data = {"dev": sentences, "train": sentences, "test": sentences}
    data = parse_conll(files)

    # clean and tokenize the data
    train_data = clean_data(data['train'], lang)
    dev_data = clean_data(data['dev'], lang)
    test_data = clean_data(data['test'], lang)

    return train_data, dev_data, test_data

