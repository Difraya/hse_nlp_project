import nltk

def download_nltk_resources():
    resources = ['punkt', 'stopwords']
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource)

download_nltk_resources()
