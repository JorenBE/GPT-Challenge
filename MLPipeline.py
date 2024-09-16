import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

import optuna

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from loguru import logger

optuna.logging.set_verbosity(optuna.logging.INFO)

#! Not very sure if the implementation of optuna for multi target modelling is ok??
#! The NaN values are replaced by -1, is there a better way?
# TODO: Make it better by first splitting, then converting the smiles to the fingerprints. This way usually, less computation for the conversion is needed.

Regression_models = [
    "LinearRegression",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "LogisticRegression"
]

Classification_models = [
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "XGBClassifier",
    "KNeighborsClassifier",
    "MLPClassifier"
]


@dataclass
class MLmodel:
    modelType: str
    df: pd.DataFrame
    target: List[str]
    features: List[str]
    feature_types: List[str]
    target_type: List[str] = field(default_factory=lambda: ['binary'])
    test_count: int = 50
    train_count: int = 50
    randomSeed: int = 42
    hyperparameter_tuning: bool = False
    param_grid: Optional[Dict[str, Any]] = None
    cv: int = 5
    optimization_method: str = 'grid_search'

    model: Optional[BaseEstimator] = field(init=False, default=None)
    X_train: Optional[pd.DataFrame] = field(init=False, default=None)
    X_val: Optional[pd.DataFrame] = field(init=False, default=None)
    X_test: Optional[pd.DataFrame] = field(init=False, default=None)
    y_train: Optional[pd.Series] = field(init=False, default=None)
    y_val: Optional[pd.Series] = field(init=False, default=None)
    y_test: Optional[pd.Series] = field(init=False, default=None)
    objective: Optional[Callable] = None

    def __post_init__(self):

        assert all([f in self.df.columns for f in self.target]), \
            "One or more target column not found in the DataFrame."

        assert all([f in self.df.columns for f in self.features]), \
            "One or more feature columns not found in the DataFrame."

        assert len(self.features) == len(self.feature_types), \
            "Number of feature types should match the number of features."

        assert all([f in ['SMILES', 'numerical'] for f in self.feature_types]), \
            "Feature types should be either 'SMILES' or 'numerical'."

        self.df = self.df.fillna(-1)  # Ensure no missing values
        self.model = None  # This will hold the instantiated model

        if 'SMILES' in self.feature_types:
            self.df['fingerprints'] = self.df['SMILES'].apply(self.smiles_to_fingerprint)
            self.features.pop(self.features.index('SMILES'))
            self.df = self.df.dropna(subset=['fingerprints'])
            self.X = self.df[self.features + ['fingerprints']]
        else:
            self.X = self.df[self.features]

        self.y = self.df[self.target]

        total_samples = len(self.X)

        # Check if the required number of train and test samples are available
        if self.train_count + self.test_count > total_samples:
            raise ValueError("Requested train_count and test_count exceed the available data points.")

        # Split to get the exact number of train samples
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(
            self.X, self.y,
            train_size=self.train_count,
            random_state=self.randomSeed,
            stratify=self.y if self.target_type == 'binary' else None
        )

        # Split remaining data to get the exact number of test samples
        self.X_test, _, self.y_test, _ = train_test_split(
            X_temp, y_temp,
            train_size=self.test_count,
            random_state=self.randomSeed,
            stratify=y_temp if self.target_type == 'binary' else None
        )

        # Ensure X_train is a 2D NumPy array
        data_to_convert = ['X_train', 'X_test', 'y_train', 'y_test']
        for attr in data_to_convert:
            setattr(self, attr, getattr(self, attr).values)
        self.X_train = np.squeeze(self.X_train)
        self.X_test = np.squeeze(self.X_test)
        self.X_train = np.array([np.array(x).flatten() for x in self.X_train])
        self.X_test = np.array([np.array(x).flatten() for x in self.X_test])
        # Not optimal
        if len(self.target) < 2:
            self.y_train = self.y_train.ravel()
            self.y_test = self.y_test.ravel()

        logger.info('ndim y_train: {}', self.y_train.ndim)
        logger.info('ndim x_train: {}', self.X_train.ndim)
        logger.info('shape y_train: {}', self.y_train.shape)
        logger.info('shape x_train: {}', self.X_train.shape)

    # Initialize the model based on modelType
    # Regression models
        if self.modelType == "LinearRegression":
            self.model = LinearRegression()
        elif self.modelType == "DecisionTreeRegressor":
            self.model = DecisionTreeRegressor()
        elif self.modelType == "RandomForestRegressor":
            self.model = RandomForestRegressor()
        elif self.modelType == "LogisticRegression":
            self.model = LogisticRegression()
        # Classification models
        elif self.modelType == "DecisionTreeClassifier":
            self.model = DecisionTreeClassifier()
        elif self.modelType == "RandomForestClassifier":
            self.model = RandomForestClassifier()
        elif self.modelType == "GradientBoostingClassifier":
            self.model = GradientBoostingClassifier()
        elif self.modelType == "XGBClassifier":
            self.model = XGBClassifier()
        elif self.modelType == "KNeighborsClassifier":
            self.model = KNeighborsClassifier()
        elif self.modelType == "MLPClassifier":
            self.model = MLPClassifier()
        else:
            raise ValueError(f"Model type {self.modelType} is not supported.")

        if len(self.target) >= 2 and self.modelType in Classification_models:
            self.model = MultiOutputClassifier(self.model)
        elif len(self.target) >= 2 and self.modelType in Regression_models:
            self.model = MultiOutputRegressor(self.model)

    def smiles_to_fingerprint(self, smiles) -> np.ndarray:
        """
        Convert a SMILES string to a molecular fingerprint using RDKit.
        """
        # smiles = row[self.features[0]]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # print(f"Invalid SMILES string: {smiles}, returning None.")
            return None
        # Generate Morgan fingerprint
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=20, fpSize=512)
        fingerprint = mfpgen.GetFingerprint(mol)

        return np.array(fingerprint)

    def hyperparameter_optimization(self):
        """
        Perform hyperparameter optimization.
        """
        if self.optimization_method == 'grid_search':
            grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=self.cv,
                                       scoring='accuracy' if self.modelType in Classification_models
                                       else 'neg_mean_squared_error')

            grid_search.fit(self.X_train, self.y_train)
            print(f"Best hyperparameters: {grid_search.best_params_}")
            self.model = grid_search.best_estimator_
            print(f"Best {self.modelType} model trained successfully.")

        elif self.optimization_method == 'random_search':
            raise NotImplementedError("Random search is not implemented yet.")

        elif self.optimization_method == 'optuna':
            study = optuna.create_study(direction='maximize' if self.modelType in Classification_models else 'minimize')
            study.optimize(self.objective, n_trials=50)
            best_params = study.best_params

            if len(self.target) < 2:
                self.model.set_params(**best_params)
            else:
                self.model.estimator.set_params(**best_params)

            best_model = self.model.fit(self.X_train, self.y_train)
            self.model = best_model
            print(f"Best {self.modelType} model trained successfully with "
                  "hyperparameter tuning using Optuna.")
            print(f"Best hyperparameters: {best_params}")

        else:
            raise NotImplementedError(f"The optimization method is not implemented yet: {self.optimization_method}")

    def train(self):
        if self.model is None:
            raise ValueError("No model is defined.")

        # Check if hyperparameter tuning is requested
        if self.hyperparameter_tuning:
            if self.optimization_method == 'grid_search' and self.param_grid is None:
                raise ValueError("Parameter grid is required for hyperparameter tuning.")
            if self.optimization_method == 'optuna' and self.objective is None:
                raise ValueError("Objective function is required for Optuna optimization.")
            if self.optimization_method not in ['grid_search', 'optuna']:
                raise ValueError(f"Invalid optimization method: {self.optimization_method}. "
                                 f"Choose from 'grid_search', 'optuna'.")
            self.hyperparameter_optimization()
        else:
            # Train the model without hyperparameter tuning
            self.model.fit(self.X_train, self.y_train)

        print(f"{self.modelType} model trained successfully.")

    def predict(self, X=None):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        if X is None:
            X = self.X_test  # Default to test set if no data is provided

        # Predict using the trained model
        predictions = self.model.predict(X)
        return predictions

    def evaluate(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        predictions = self.predict(self.X_test)

        if self.modelType in Regression_models:
            # Predict on the test set
            eval = mean_squared_error(self.y_test, predictions)
            print(f"Mean Squared Error for {self.modelType}: {eval:.2f}")
        else:
            if len(self.y_test.shape) >= 2:
                # Multiple target columns
                accuracies = [accuracy_score(self.y_test[:, i], predictions[:, i]) for i in range(self.y_test.shape[1])]
                print(f"Accuracies for each target in {self.modelType}: {accuracies}")
                eval = accuracies
            else:
                # Predict on the test set
                eval = accuracy_score(self.y_test, predictions)
                print(f"Accuracy for {self.modelType}: {eval:.2f}")
        return eval

    def getValues(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def getdfAnalysis(self, orginal_df: pd.DataFrame):
        number_of_samples = len(orginal_df)
        number_of_wrong_smiles = len(orginal_df) - len(self.df)
        clean_df = self.df
        return number_of_samples, number_of_wrong_smiles, clean_df


@dataclass
class BinTheTarget:
    df: pd.DataFrame
    target: str
    bins: int = 2

    def __post_init__(self):
        # Use pd.qcut to bin data into quantiles
        self.df[self.target] = pd.qcut(self.df[self.target], q=self.bins, labels=False)
        self.df = self.df.dropna(subset=[self.target])
