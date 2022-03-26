"""
To run:
python -m models.pysentimiento_feats --file-name labels_racism
python -m models.pysentimiento_feats --file-name evaluation_public
python -m models.pysentimiento_feats --file-name evaluation_final

"""
import argparse
from pathlib import Path

import pandas as pd
from pysentimiento import create_analyzer
from pysentimiento.preprocessing import preprocess_tweet
from tqdm import tqdm
from utils.analyzer_utils import get_analyzer_features

data_path = Path("data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate pysentimiento features"
    )
    parser.add_argument("--file-name", type=str, required=True)
    args = parser.parse_args()
    file_name = args.file_name

    df = pd.read_csv(data_path / f"{file_name}.csv", delimiter="|")
    df["clean_message"] = df["message"].apply(preprocess_tweet)

    """Load hate speech analyzer and get features for tweet text"""
    # %% All features
    feats_dict = {}
    for task in tqdm(["hate_speech", "emotion", "sentiment"]):
        analyzer = create_analyzer(task=task, lang="es")
        feats_dict[task] = get_analyzer_features(df.message, analyzer)
        feats_dict[f"{task}_clean"] = get_analyzer_features(
            df.clean_message, analyzer)
        cols = feats_dict[f"{task}_clean"].columns
        feats_dict[f"{task}_clean"].columns = [f"{col}_clean" for col in cols]

    all_feats = pd.concat(feats_dict.values(), axis=1)
    final_df = pd.concat([df, all_feats], axis=1)

    # %% Save data
    (
        final_df
        .to_csv(data_path / f"pysentimiento_{file_name}.csv", index=False, float_format="%.3f")
    )
