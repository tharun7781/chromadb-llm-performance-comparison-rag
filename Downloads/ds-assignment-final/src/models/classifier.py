from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
# *** NO LONGER NEEDED: from src.utils.store import AssignmentStore ***

# ADDED IMPORTS FOR EVALUATION
from sklearn.metrics import roc_auc_score, accuracy_score


class Classifier(ABC):
    @abstractmethod
    def train(self, *params) -> None:
        pass

    @abstractmethod
    def evaluate(self, *params) -> Dict[str, float]:
        pass

    @abstractmethod
    def predict(self, *params) -> np.ndarray:
        pass


class SklearnClassifier(Classifier):
    def __init__(
        self, estimator: BaseEstimator, features: List[str], target: str,
    ):
        self.clf = estimator
        self.features = features
        self.target = target

    def train(self, df_train: pd.DataFrame):
        self.clf.fit(df_train[self.features].values, df_train[self.target].values)

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, float]:
        # Implementation to fix NotImplementedError
        X_test = df_test[self.features].values
        y_true = df_test[self.target].values
        y_pred_proba = self.clf.predict_proba(X_test)[:, 1] 
        
        metrics = {}
        
        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        else:
            metrics["roc_auc"] = 0.5 

        y_pred_class = self.clf.predict(X_test)
        metrics["accuracy"] = accuracy_score(y_true, y_pred_class)
        
        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(df[self.features].values)[:, 1]