import string

def preprocess_text(text):
    """
    This function normalises text for later use in
    a machine learning pipeline
    """
    if isinstance(text, str):
        text = text.lower()
        text = text.translate(str.maketrans('','', string.punctuation))

    return ' '.join(text.split())
