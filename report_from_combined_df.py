import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, roc_auc_score
from dataclasses import dataclass
from typing import Any
from functools import partial
import argparse


@dataclass
class Metric:
    name: str
    fn: Any

metrics = [
    Metric(name="precision", fn=lambda y_true, y_pred, y_pred_proba: accuracy_score(y_true, y_pred)),
    Metric(name="accuracy", fn=lambda y_true, y_pred, y_pred_proba: precision_score(y_true, y_pred)),
    Metric(name="roc_auc", fn=lambda y_true, y_pred, y_pred_proba: roc_auc_score(y_true, y_pred_proba)),
]

def read_csv(csv_file: str) -> pd.DataFrame:

    # First find all the columns and just select which columns we don't need
    columns = pd.read_csv(csv_file, nrows=1).columns
    columns = [str(c) for c in columns]
    columns = [c for c in columns if not c.startswith('exclude')]
    columns = [c for c in columns if not c.startswith('Unnamed:')]
    columns = [c for c in columns if c != 'filename']
    df = pd.read_csv(csv_file, usecols=columns)
    
    return df


def create_report(csv_file: str):

    df = read_csv(csv_file)
    # For every feaure set, run names start with 0 and end with 255. It
    # becomes easier to aggregate if run names were unique
    # Make them unique by simply appending the run name and feature set name
    df['run_name'] = df['feature_set'] + ':' + df['run_name']
            
    def calculate_metrics(group):
        return pd.Series({m.name: m.fn(group['y_true'], group['y_pred'], group['y_pred_proba']) for m in metrics})

    metrics_df = df.groupby('run_name').apply(calculate_metrics)
    return metrics_df


def main() -> None:
    parser = argparse.ArgumentParser(prog = "GenReport", 
                                     description="From a combined dataframe, create a report")
    parser.add_argument("-f", "--file", type=str, required=True)
    args = parser.parse_args()
    print(create_report(args.file))


if "__main__" == __name__:
    main()