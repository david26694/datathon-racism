import re


def camel_case_split(identifier):
    matches = re.finditer(
        '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return " ".join([m.group(0) for m in matches])


def transform_hashtag(x):
    """Remove hashtags and transform CamelCase to underscore"""
    x = camel_case_split(x)
    x = x.replace("#", "")
    return x


def preprocess_line(text):
    """http and @user to http_token and user_token"""
    new_text = []
    for t in text.split(" "):
        t = 'usuario' if t.startswith('@') and len(t) > 1 else t
        t = 'web' if t.startswith('http') else t
        new_text.append(t)
    joined = " ".join(new_text)
    return joined.lower()


def process_tweet(x):
    """Remove hashtags and transform CamelCase to underscore"""
    x = transform_hashtag(x)
    x = preprocess_line(x)
    return x


if __name__ == "__main__":
    print(process_tweet(
        "Hola #DonaldTrum #Trump @donald #ArribaVOXVenceremos #VoxExtremaNecesidad #SiguemeYTeSigoVOX"))