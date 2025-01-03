import nltk

def download_nltk_resources():
    resources = [''punkt_tab'', 'stopwords', 'averaged_perceptron_tagger_eng']
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource)

download_nltk_resources()
