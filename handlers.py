import os
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score,
                             recall_score, precision_score)
from typing import Union
from tableshift import get_dataset
from scipy.stats import wilcoxon

import logging
logging.basicConfig(
    level=logging.INFO,  # Sets the minimum log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log output format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date/time format
)

def info(a:str):
    logging.info(a)


LIGHTWEIGHT_DATASETS = [
    [ 'Childhood Lead',          'nhanes_lead'             ],
    [ 'FICO HELOC',              'heloc'                   ],
    [ 'Hospital Readmission',    'diabetes_readmission'    ],
    [ 'Voting',                  'anes'                    ],
]

MIDWEIGHT_DATASETS = [
    [ 'College Scorecard',       'college_scorecard'       ],
    [ 'Hypertension',            'brfss_blood_pressure'    ],
]

HEAVYWEIGHT_DATASETS = [
    [ 'ASSISTments',             'assistments'             ],
    [ 'Diabetes',                'brfss_diabetes'          ],
    [ 'Food Stamps',             'acsfoodstamps'           ],
    [ 'ICU Length of Stay',      'mimic_extract_los_3'     ],
    [ 'ICU Mortality',           'mimic_extract_mort_hosp' ],
    [ 'Income',                  'acsincome'               ],
    [ 'Public Health Insurance', 'acspubcov'               ],
    [ 'Sepsis',                  'physionet'               ],
    [ 'Unemployment',            'acsunemployment'         ],
]

AVALIABLE_DATASETS = sorted(
    LIGHTWEIGHT_DATASETS + MIDWEIGHT_DATASETS + HEAVYWEIGHT_DATASETS, 
    key=lambda x: x[0]
)

class DatasetHandler:
    """
    Handles dataset loading and preparation for training and testing.

    This class supports two main types of datasets:
    1. Datasets stored locally in Feather format, where the last column is the target.
    2. Datasets retrieved using the TableShift library.

    Parameters
    ----------
    base_path : str
        Path to the directory where local dataset files (.feather) are stored.
    """
    def __init__(self, base_path):
        self.base_path = base_path

    # def load_data(self, dataset_name, partition):
    #     """
    #     Load a dataset from Feather files.

    #     The method expects the target variable to be the last column of the dataset.
    #     When a list is provided, the corresponding partitions are concatenated.

    #     Parameters
    #     ----------
    #     dataset_name : str
    #         Name prefix of the dataset file(s).
    #     partition : str or list of str
    #         Partition name(s) to load (e.g., 'train', 'test', or ['train', 'val']).

    #     Returns
    #     -------
    #     X : np.ndarray
    #         Feature matrix.
    #     y : np.ndarray
    #         Target vector.
    #     labels : np.ndarray
    #         Names of the feature columns.
    #     """
    #     if isinstance(partition, str):
    #         filename = f"{dataset_name}_{partition}.feather"
    #         file_path = os.path.join(self.base_path, filename)
    #         df = pd.read_feather(file_path)
    #         X = df.iloc[:, :-1]
    #         y = df.iloc[:, -1]
    #     elif isinstance(partition, list):
    #         df = pd.DataFrame()
    #         for p in partition:
    #             filename = f"{dataset_name}_{p}.feather"
    #             file_path = os.path.join(self.base_path, filename)
    #             df = pd.concat([df,pd.read_feather(file_path)])
    #     else:
    #         raise ValueError("The partition must be a string or a list of strings")
    #     X = df.iloc[:, :-1]
    #     y = df.iloc[:, -1]
    #     labels = np.array(df.columns)[:-1]
    #     return X.values, y.values, labels
    
    def load_data(self, dataset_name, partition):
        """
        Load a dataset using the TableShift library.

        This method uses the `get_dataset` function to retrieve a dataset from cache or download it.
        The target variable is assumed to be included in the returned data.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset as recognized by TableShift.
        partition : str
            Partition name to load (e.g., 'train', 'test', 'ood_test').

        Returns
        -------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target vector.
        """
        dset = get_dataset(
            dataset_name, 
            cache_dir=self.base_path, 
            use_cached=True
        )
        
        X, y, _, _ = dset.get_pandas(partition)
        return X.values, y.values


class ModelTrainer:
    """
    Trains models and evaluates their performance.
    Allows extra model parameters at initialization, which are filtered per model.
    """
    def __init__(self, unbiased_model, biased_model, *, n_jobs=-1, max_samples:Union[float, int, None]=1.0, n_estimators=1000, 
                 max_features="sqrt", random_state=2, **model_kwargs):
        self.params = {
            "n_jobs": n_jobs,
            "max_samples": max_samples,
            "n_estimators": n_estimators,
            "max_features": max_features,
            "random_state": random_state,
        }
        # It stores any other argument the user wants to try to pass.
        self.extra_params = model_kwargs
        self.unbiased_model = unbiased_model
        self.biased_model = biased_model

    def _safe_model_init(self, model_class, **override_kwargs):
        """
        Initialize the given model class, using only parameters valid for it.

        Parameters
        ----------
        model_class : type
            The model class (e.g., RandomForestClassifier).
        override_kwargs : dict
            Parameters to override default + extra ones.

        Returns
        -------
        model : instance
            Instantiated model with valid parameters.
        """
        all_kwargs = {**self.params, **self.extra_params, **override_kwargs}
        valid_keys = model_class().get_params().keys()
        filtered_kwargs = {k: v for k, v in all_kwargs.items() if k in valid_keys}
        return model_class(**filtered_kwargs)

    def train_base_model(self, X, y, **kwargs):
        model = self._safe_model_init(self.unbiased_model, **kwargs)
        model.fit(X, y)
        return model

    def train_adapted_model(self, X, y, feature_bias, **kwargs):
        model = self._safe_model_init(self.biased_model, **kwargs)
        model.fit(X, y, feature_bias=feature_bias)
        return model

class Evaluation:
    """
    Evaluates models and saves results.
    """
    def __init__(self, filename, zero_division=0, average="macro", datasets=None):
        self.results = pd.DataFrame(datasets, 
                                    columns=["dataset", "experiment"]
                                    ).set_index("experiment")
        self.filename = filename
        self.zero_division = zero_division
        self.average = average

    def save_results(self):
        self.results.to_markdown(f"{self.filename}.md")
        self.results.to_csv(f"{self.filename}.csv")

    def load_results(self):
        self.results = pd.read_csv(f"{self.filename}.csv")

    def update_results(self, experiment, metric, value):
        self.results.loc[experiment, metric] = value
        self.save_results()

    def compute_metrics(self, y_true, y_pred):
        kwargs = {
            "zero_division": self.zero_division, 
            "average": self.average
        }
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, **kwargs),
            "precision": precision_score(y_true, y_pred, **kwargs),
            "recall": recall_score(y_true, y_pred, **kwargs)
        }

    def update_p_value(self, column):
        n = self.results.shape[0]
        self.results.iloc[n,0] = "p-value"
        arr_notna = self.results.iloc[:,[1,2]].dropna(axis=0).values
        p_value = wilcoxon(arr_notna[:,1], arr_notna[:,column]).pvalue
        self.results.iloc[n,column+1] = p_value


    def generate_latex_report(self, strategy_names=None):
        """
        Generate LaTeX tables for the evaluation results.

        Args:
            strategy_names (list of str): List of strategy names to include in the report.

        Returns:
            str: Complete LaTeX content for the report.
        """
        
        basic_columns = ["dataset", "accuracy_id", "f1_id", 
                         "precision_id", "recall_id", "accuracy_ood", 
                         "f1_ood", "precision_ood", "recall_ood"]

        metrics = [m[:-3] for m in basic_columns if m[-3:] == "_id"]
        n_strategies = int((self.results.shape[1] - len(basic_columns)) / len(metrics))
        
        if strategy_names is None:
            strategy_names = ["Strategy " + f"{i+1}" for i in range(n_strategies)]
            
        latex_content = []
        
        for i in range(n_strategies):
            for metric in metrics[:2]:
                raw_string = strategy_names[i] + ", evaluated by " + metric
                latex_content.append(r"\begin{table}[h!]")
                latex_content.append(r"\centering")
                latex_content.append(r"\caption{" + raw_string + r"}")
                latex_content.append(r"\begin{tabular}{lc|cc}")
                latex_content.append(r"\toprule")
                
                latex_content.append(r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{ID} & \multicolumn{2}{c}{OOD} \\")
                latex_content.append(r"& & {baseline} & {biased} \\")
                latex_content.append(r"\midrule")
                cols = ["dataset"]
                cols.append(f"{metric}_id")
                cols.append(f"{metric}_ood")
                cols.append(f"{metric}_s{i+1}")
                df_ev = self.results[cols].copy()
                df_ev[cols[1:]] *= 100
                df_styled = df_ev.style.hide_index()
                df_styled = df_styled.highlight_max(
                    subset=cols[2:], axis=1, props="textbf:--rwrap;"
                    )
                df_styled = df_styled.format("{:.2f}", subset=cols[1:], na_rep="N/A")
                lines = df_styled.to_latex().split("\n")
                for l in lines[2:-2]:
                    latex_content.append(l)
                latex_content.append(r"\bottomrule")
                df_notna = df_ev.dropna()    
                p_value = wilcoxon(df_notna.iloc[:,-2], df_notna.iloc[:,-1]).pvalue
                mc_start = r"\multicolumn{4}{l}{pvalue: "
                mc_end = r"}"
                latex_content.append(mc_start + f"{p_value:0.3f}" + mc_end)
                latex_content.append(lines[-2])
                latex_content.append(r"\end{table}")
                latex_content.append("")
        
        print("\n".join(latex_content))
        return "\n".join(latex_content)

