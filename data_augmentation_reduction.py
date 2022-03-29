from pathlib import Path
import pandas as pd
import re
import random

data_path = Path("data")
input_path = data_path / "split"
data_path = Path("data")

# %%
a = ["74 inmigrantes ilegales de Canarias vuelan a Alicante pero 60 quedan libres por el colapso del CETI",
     "@muna_mnshira la que va repartiendo amor y prejuzga a la gente sin conocer nada de su vida. que fácil me resulta quintaros la careta “ainmigrantes negrossa"]

words = pd.read_csv('predatathon/data/racist_words.txt', header = None)[0].to_list()

def augmentation(words, train_df):
    """replicate strong messages with other words"""
    many_labels = (
        train_df
            .groupby("message", as_index=False)
            .agg(
            racists=('label', lambda x: (x == 'racist').sum()),
            non_racists=('label', lambda x: (x == 'non-racist').sum()),
            unknowns=('label', lambda x: (x == 'unknown').sum()),
        )
            .assign(
            some_racist=lambda x: x.racists > 0,
            some_non_racist=lambda x: x.non_racists > 0,
            some_unknown=lambda x: x.unknowns > 0,
        )
    )

    many_labels['total_reviews'] = many_labels['racists'] + many_labels['non_racists'] + many_labels['unknowns']
    cut = many_labels.total_reviews.quantile(0.80)
    strong = many_labels.query('total_reviews > @cut & abs(racists - non_racists) == total_reviews')


    # replicate strong messages with other words
    strong['message'] = strong.message.str.lower().apply(lambda x: re.sub(pattern="|".join(words), repl= random.sample(words, 1)[0], string=x))
    strong['label'] = 'racist'
    strong.loc[strong['racists'] < strong['non_racists'], 'label'] = 'non-racist'
    add = strong[['message', 'label']]
    train_df = train_df.append(add)


    return train_df


# %%
def remove_weak(train_df):
    """remove inconsistent messages"""
    many_labels = (
        train_df
            .groupby("message", as_index=False)
            .agg(
            racists=('label', lambda x: (x == 'racist').sum()),
            non_racists=('label', lambda x: (x == 'non-racist').sum()),
            unknowns=('label', lambda x: (x == 'unknown').sum()),
        )
            .assign(
            some_racist=lambda x: x.racists > 0,
            some_non_racist=lambda x: x.non_racists > 0,
            some_unknown=lambda x: x.unknowns > 0,
        )
    )

    many_labels['total_reviews'] = many_labels['racists'] + many_labels['non_racists'] + many_labels['unknowns']


    cut = many_labels.total_reviews.quantile(0.80)
    weak = many_labels.query('total_reviews > @cut & racists > 1 & non_racists > 1')
    train_df = train_df[~train_df.message.isin(weak['message'])]

    return train_df


# %%
tr = pd.read_csv(input_path / "labels_racism_train.txt", delimiter="|")

augmentation(words=words, train_df=tr)

# %%

remove_weak(tr)