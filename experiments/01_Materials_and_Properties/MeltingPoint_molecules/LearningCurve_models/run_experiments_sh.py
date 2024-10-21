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

MODEL_FOLDER_MAP = {
    'EleutherAI/gpt-j-6b': 'gptj',
    'meta-llama/Meta-Llama-3.1-8B-Instruct': 'llama',
    'meta-llama/Meta-Llama-3.1-8B': 'llama3.1',
    'mistralai/Mistral-7B-Instruct-v0.3': 'mistral',
    'meta-llama/Llama-3.2-1B': 'llama3.2-1B',
}

TARGET_PROPERTY_MAP = {
    'E_coh_bin': 'Cohesive Energy',
    'T_g_bin': 'Glass transition Temperature',
    'R_gyr_bin': 'Squared Radius of gyration',
    'Densities_bin' : 'Density',
    'viscosity_2bins_5050': 'viscosity',
    'mp_bin': 'Melting point'
}

def train_test(train_size: int = 300, random_state: int = 42, model = 'EleutherAI/gpt-j-6b', num_epochs = 30, representation = 'SMILES', target = 'E_bin', data_file = 'train_polymers.csv', max_test_data = 50):
    try:
        model_map = MODEL_FOLDER_MAP[model]
        out_folder = f'out_{model_map}'
        pred_folder = f'predictions_{model_map}'
    except:
        out_folder = 'out'
        pred_folder = 'predictions'

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder)

    df = pd.read_csv(data_file)

    data_summary = {
        'datafile':data_file,
        'target':target,
        'representation' : representation
    }

    try:
        property_name = TARGET_PROPERTY_MAP[target]
    except:
        property_name = 'property'
        
    config = {
        "property_name": property_name,
        "tune_settings": {"num_train_epochs": num_epochs, "learning_rate": 3e-4},
        "tokenizer_kwargs": {"cutoff_len": 128},
        "base_model": model,
        "batch_size": 2,
        "inference_batch_size": 2,
    }


    wandb.init(
        # set the wandb project where this run will be logged
        project="gpt-challenge-classif-polymer",
        # track hyperparameters and run metadata
        config={
            "model": model,
            "target": target,
            **config,
            "train_size": train_size,
            "num_epochs": num_epochs,
        },
        tags=["classification", "rate", model],
    )


    df = df.dropna(subset=[representation, target])
    
    df = equalize_dataset(df, target)

    df_train, df_test = train_test_split(
        df,
        train_size=train_size,
        test_size= min(len(df)-train_size, max_test_data),
        random_state=random_state,
        stratify=df[target].astype(int).values,
    )

    print(len(df_train), len(df_test))

    classifier = PEFTClassifier(
        **config,
    )

    classifier.fit(df_train[representation], df_train[target])


    predictions =[]
    test_batch = 5
    for i in range(0, len(df_test[representation]), test_batch):
        preds = classifier._predict(df_test[representation].iloc[i:i+test_batch])
        preds = np.array(preds[0]).astype(int)
        predictions.extend(preds)
    print(f"predictions {predictions}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    #predictions = classifier._predict(df_test[representation])

    predictions = np.array(predictions)

    #predictions = np.array(predictions[0])

    df_test['prediction'] = predictions
    df_test['partition'] = 'test'
    df_test = df_test[[representation, target, 'partition', 'prediction']]

    df_train['prediction'] = [999] * len(df_train)
    df_train['partition'] = 'train'
    df_train = df_train[[representation, target, 'partition', 'prediction']]

    df_all = pd.concat([df_train, df_test])
    df_all.to_csv(os.path.join(pred_folder, f'{timestamp}_predictions_seed{random_state}_{target}_{representation}_{num_epochs}epoch.csv'))

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
        os.path.join(out_folder, f"{timestamp}_{train_size}_{num_epochs}_predictions.pkl"),
        {"results":results, "predictions": predictions, "true": true, "train_size": train_size, "config": config, 'data_summary':data_summary,},
    )

if __name__ == "__main__":
    fire.Fire(train_test)
