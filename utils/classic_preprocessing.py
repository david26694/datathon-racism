import pandas as pd


def preprocess_line(text):
    """http and @user to http_token and user_token"""
    new_text = []
    for t in text.split(" "):
        t = 'user_token' if t.startswith('@') and len(t) > 1 else t
        t = 'http_token' if t.startswith('http') else t
        new_text.append(t)
    joined = " ".join(new_text)
    return joined.lower()


def preprocess_str(x: pd.Series, remove_words: list) -> pd.Series:
    """
    Preprocess a string
    """
    regex_remove = r'\b(?:{})\b'.format('|'.join(remove_words))

    # To lower
    x = x.str.lower()
    # Remove stop words
    x = x.str.replace(regex_remove, '')
    # Non-alphanumeric characters
    x = x.str.replace(r'[^\w\s]', '')
    # Accents
    x = x.str.normalize('NFKD').str.encode(
        'ascii', errors='ignore').str.decode('utf-8')
    # Space normalize
    x = x.str.replace(r"\s+", " ")

    return x


def process_text(df, stop_words, text="message"):
    df = df.copy()

    new_text = df[text].apply(preprocess_line)
    new_text = preprocess_str(new_text, stop_words)
    df["processed_msg"] = new_text

    return df
