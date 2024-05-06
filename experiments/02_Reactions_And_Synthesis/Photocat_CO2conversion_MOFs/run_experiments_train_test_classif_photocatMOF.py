import os
import time

import fire
import numpy as np
import pandas as pd
import wandb
from fastcore.xtras import save_pickle
from gptjchem.peftclassifier import PEFTClassifier
from gptchem.evaluator import evaluate_classification
from pycm import ConfusionMatrix
from sklearn.model_selection import train_test_split


def train_test(train_size: int = 300, random_state: int = 42):
    if not os.path.exists("out"):
        os.makedirs("out")

    df = pd.read_csv("PhotocatCO2conversionMOFs_dataset.csv", sep=';')
    target = "activity_2bins_5050"
    representation = "catalyst_smiles"
    num_epochs = 100

    config = {
        "property_name": "BET surface area (m2/g)",
        "tune_settings": {"num_train_epochs": num_epochs, "learning_rate": 1e-3},
        "tokenizer_kwargs": {"cutoff_len": 128},
        "base_model": "EleutherAI/gpt-j-6b",
        "batch_size": 64,
        "inference_batch_size": 64,
    }


    wandb.init(
        # set the wandb project where this run will be logged
        project="gpt-challenge-classif-mofphotocat",
        # track hyperparameters and run metadata
        config={
            "model": "gpt-j-6b",
            "target": target,
            **config,
            "train_size": train_size,
            "num_epochs": num_epochs,
        },
        tags=["classification", "mofphotocat", "gpt-j-6b"],
    )


    df = df.dropna(subset=[representation, target])
    
    df_train, df_test = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        stratify=df[target].astype(int).values,
    )

    print(len(df_train), len(df_test))

    classifier = PEFTClassifier(
        **config,
    )

    classifier.fit(df_train[representation], df_train[target])

    predictions = classifier._predict(df_test[representation])

    predictions = np.array(predictions[0])
    # predictions 0 is a list, build a mask to filter out the nans
    nan_prediction_mask = np.isnan(predictions)
    num_nan_predictions = nan_prediction_mask.sum()

    print(predictions)

    true = df_test[target][~nan_prediction_mask].astype(int).values

    results = evaluate_classification(true, predictions[~nan_prediction_mask])
    results_w_float_values = {k: v for k, v in results.items() if isinstance(v, float)}
    print(f"results: {results}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    wandb.log(results_w_float_values)
    wandb.log({f"num_nan_predictions": num_nan_predictions})

    save_pickle(
        os.path.join("out", f"{timestamp}_{train_size}_{random_state}_predictions.pkl"),
        {"predictions": predictions, "true": true, "train_size": train_size, "config": config},
    )

if __name__ == "__main__":
    fire.Fire(train_test)