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

def equalize_dataset(df, col, verbose = True):
    '''Create a balanced dataset by undersampling the majority class'''
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

DATA_FILE = os.path.join('data', 'polymers_formattedData.csv')
MAX_TEST_DATA = 50

def train_test(train_size: int = 300, random_state: int = 42, model = 'EleutherAI/gpt-j-6b', num_epochs = 30, representation = 'SMILES', target = 'E_bin'):
    
    #create folder where output will be saved
    if not os.path.exists("out"):
        os.makedirs("out")

    # read data file and define 'target' and 'representation' columns
    df = pd.read_csv(DATA_FILE)
    target = target
    representation = representation

    # create dict for logging
    data_summary = {
        'datafile':DATA_FILE,
        'target':target,
        'representation' : representation
    }

    # define experiment parameters
    config = {
        "property_name": "Adhesive Free Energy",
        "tune_settings": {"num_train_epochs": num_epochs,
                           "learning_rate": 3e-4}
                           ,
        "tokenizer_kwargs": {"cutoff_len": 128},
        "base_model": model,
        "batch_size": 10,
        "inference_batch_size": 2,
    }

    # setup wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project="gpt-challenge-classif-polymers",
        # track hyperparameters and run metadata
        config={
            "model": model,
            "target": target,
            **config,
            "train_size": train_size,
            "num_epochs": num_epochs,
        },
        tags=["classification", DATA_FILE, model],
    )

    # drop entries with no data
    df = df.dropna(subset=[representation, target])
    
    #if needed, create balanced dataset based on 'target' by undersampling the majority class
    df = equalize_dataset(df, target)

    # create train and test set
    df_train, df_test = train_test_split(
        df,
        train_size=train_size, # training set size defined as parameter
        test_size= min(len(df)-train_size, MAX_TEST_DATA),
        random_state=random_state,
        stratify=df[target].astype(int).values,
    )

    # initiate Classifier Model
    classifier = PEFTClassifier(
        **config,
    )

    # fit training data
    classifier.fit(df_train[representation], df_train[target])
    
    # Use model to predict test data
    pred_batch = 5
    predictions =[]
    for i in range(0, len(df_test[representation]), pred_batch):
        preds = classifier._predict(df_test[representation].iloc[i:i+pred_batch])
        preds = np.array(preds[0]).astype(int)
        predictions.extend(preds)
    print(f"predictions {predictions}")

    # format predictions
    predictions = np.array(predictions)
    nan_prediction_mask = np.isnan(predictions)
    num_nan_predictions = nan_prediction_mask.sum()
    true = df_test[target][~nan_prediction_mask].astype(int).values

    # calculate classifiction metrics of experiment
    results = evaluate_classification(true, predictions[~nan_prediction_mask])
    results_w_float_values = {k: v for k, v in results.items() if isinstance(v, float)}
    print(f"results: {results}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    wandb.log(results_w_float_values)
    wandb.log({f"num_nan_predictions": num_nan_predictions})

    # save results in standerized pickle file
    save_pickle(
        os.path.join("out", f"{timestamp}_{train_size}_{num_epochs}_predictions.pkl"),
        {"results":results, "predictions": predictions, "true": true, "train_size": train_size, "config": config, 'data_summary':data_summary,},
    )

if __name__ == "__main__":
    fire.Fire(train_test)
