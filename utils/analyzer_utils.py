import pandas as pd
from pysentimiento import Analyzer


def get_analyzer_features(texts: pd.Series, analyzer: Analyzer):
    """
    Get analyzer features from a pd Series
    """

    predictions = analyzer.predict(texts)

    dfs = []
    for i, prediction in enumerate(predictions):
        dfs.append(pd.DataFrame(prediction.probas, index=[i]))

    return pd.concat(dfs)
