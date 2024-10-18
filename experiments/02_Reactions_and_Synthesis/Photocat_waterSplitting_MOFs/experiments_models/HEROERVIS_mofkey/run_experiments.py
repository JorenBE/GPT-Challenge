import os

import pandas as pd
from fastcore.xtras import save_pickle
from gptchem.evaluator import evaluate_classification
from gptchem.utils import make_outdir
from sklearn.model_selection import train_test_split

from gptjchem.gptjclassifier import GPTJClassifier

DATAFILE = 'final_noDuplicates.csv'
REPRESENTATIONS = [
    'mofid',
    'mofkey'
][::-1]

TARGET = 'HER_bool'

MAX_NUM_TEST_POINTS = 25

def train_test(size, representation, seed, num_epochs, lr = 1e-4, strat = None):      
    df = pd.read_csv(DATAFILE)
    for i in [representation, 'HER_bool']:
        if not i in df.columns:
            raise KeyError('{} column not in datframe ({})'.format(i, df.columns))
        
    df = df.dropna(subset=[representation, TARGET])
    df['y'] = df[TARGET]
    print(df['y'].value_counts())

    stratified = False
    if strat != None:
        strat = df[strat]
        stratified = True

    train_df, test_df = train_test_split(df, train_size=size, test_size=min([len(df)-size, MAX_NUM_TEST_POINTS]), random_state=seed, stratify=strat)

    classifier = GPTJClassifier("HER", tune_settings={"num_epochs": num_epochs, "lr": lr}, inference_batch_size=2, inference_max_new_tokens=100)
    
    classifier.fit(train_df[representation].to_list(), train_df["y"].to_list())

    y_pred = classifier.predict(test_df[representation].to_list())

    results = evaluate_classification(test_df["y"].to_list(), y_pred)

    dirname = make_outdir('')

    res  = {
        **results,
        "size": size,
        "representation": representation,
        "num_epochs": num_epochs,
        "lr": lr,
        "seed": seed,
        "stratified":stratified
    }
    save_pickle(os.path.join(dirname, f"results_{size}_{seed}_{representation}.pkl"), res)
    print(res)
    print("Pickle saved: OK")


if __name__ == "__main__":
    for seed in range(3):
        for num_epochs in [20]:
            for lrs in [1e-4]:
                for strati in ['y']:
                    for representation in REPRESENTATIONS:
                        for size in [50][::-1]:
                            try:
                                print('#Epoch: {} ; lr {}; {}; size {} ({})'.format(num_epochs, lrs, representation, size, seed+1))
                                train_test(size, representation, seed + 14556, num_epochs, lr=lrs, strat=strati)
                            except Exception as e:
                                print(e)
                                pass
