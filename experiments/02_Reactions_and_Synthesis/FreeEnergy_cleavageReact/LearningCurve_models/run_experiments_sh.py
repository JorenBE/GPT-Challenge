import os
import time

import fire
import numpy as np
import pandas as pd
import wandb
from fastcore.xtras import save_pickle
from gptchem_llama.peftclassifier import PEFTClassifier
from gptchem.evaluator import evaluate_classification
from pycm import ConfusionMatrix
from sklearn.model_selection import train_test_split

def equalize_dataset(df, col, verbose = True):
    values_dict = dict(df[col].value_counts())
    mini = min([i for i in values_dict.values()])
    
    new_df = pd.DataFrame()
    for key in values_dict.keys():
        key_df = df.loc[df[col] == key].sample(mini)
        new_df = pd.concat([new_df, key_df])
    
    if verbose:
        new_values_dict = dict(new_df[col].value_counts())
        print(f'Balanced dataset: same number of entries ({mini}) for {new_values_dict.values()}. ({new_values_dict})')
    return new_df

DATA_FILE = 'train_ruben.csv'

MAX_TEST_DATA = 50

OUT_FOLDER = 'mistral'

def train_test(train_size: int = 300, random_state: int = 42, model = 'EleutherAI/gpt-j-6b', num_epochs = 30, representation = 'Sequence', target = 'min_plus'):
    if not os.path.exists(f"out_{OUT_FOLDER}"):
        os.makedirs(f"out_{OUT_FOLDER}")
    if not os.path.exists(f"predictions_{OUT_FOLDER}"):
        os.makedirs(f"predictions_{OUT_FOLDER}")

    df = pd.read_csv(DATA_FILE)
    target = target
    representation = representation

    data_summary = {
        'datafile':DATA_FILE,
        'target':target,
        'representation' : representation
    }

    config = {
        "property_name": "relative free energy (of intermediate 4 in a catalytic cycle)",
        "tune_settings": {"num_train_epochs": num_epochs, "learning_rate": 3e-4},
        "tokenizer_kwargs": {"cutoff_len": 128},
        "base_model": model,
        "batch_size": 10,
        "inference_batch_size": 2,
    }


    wandb.init(
        # set the wandb project where this run will be logged
        project="gpt-challenge-classif-rerun-LLAMA",
        # track hyperparameters and run metadata
        config={
            "model": model,
            "target": target,
            **config,
            "train_size": train_size,
            "num_epochs": num_epochs,
        },
        tags=["classification", DATA_FILE, model, representation, target],
    )


    df = df.dropna(subset=[representation, target])
    
    df = equalize_dataset(df, target)

    df_train, df_test = train_test_split(
        df,
        train_size=train_size,
        test_size= min(len(df)-train_size, MAX_TEST_DATA),
        random_state=random_state,
        stratify=df[target].astype(int).values,
    )

    print(len(df_train), len(df_test))

    classifier = PEFTClassifier(
        **config,
    )

    classifier.fit(df_train[representation], df_train[target])
    
    pred_batch = 5
    predictions =[]
    for i in range(0, len(df_test[representation]), pred_batch):
        preds = classifier._predict(df_test[representation].iloc[i:i+pred_batch])
        preds = np.array(preds[0]).astype(int)
        predictions.extend(preds)
    print(f"predictions {predictions}")

    #predictions = classifier._predict(df_test[representation])

    predictions = np.array(predictions)

    df_test['prediction'] = predictions
    df_test['partition'] = 'test'
    df_test = df_test[[representation, target, 'partition', 'prediction']]

    df_train['prediction'] = [999] * len(df_train)
    df_train['partition'] = 'train'
    df_train = df_train[[representation, target, 'partition', 'prediction']]
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    df_all = pd.concat([df_train, df_test])
    df_all.to_csv(os.path.join(f'predictions_{OUT_FOLDER}', f'{timestamp}_predictions_seed{random_state}_{target}_{representation}_{num_epochs}epoch_{train_size}size.csv'))

    #predictions = np.array(predictions[0])

    nan_prediction_mask = np.isnan(predictions)
    num_nan_predictions = nan_prediction_mask.sum()

    true = df_test[target][~nan_prediction_mask].astype(int).values

    results = evaluate_classification(true, predictions[~nan_prediction_mask])
    results_w_float_values = {k: v for k, v in results.items() if isinstance(v, float)}
    print(f"results: {results}")

    wandb.log(results_w_float_values)
    wandb.log({f"num_nan_predictions": num_nan_predictions})

    save_pickle(
        os.path.join(f"out_{OUT_FOLDER}", f"{timestamp}_{train_size}_{num_epochs}_predictions.pkl"),
        {"results":results, "predictions": predictions, "true": true, "train_size": train_size, "config": config, 'data_summary':data_summary,},
    )

if __name__ == "__main__":
    fire.Fire(train_test)
