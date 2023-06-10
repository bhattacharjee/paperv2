import argparse
from dataclasses import dataclass
from functools import partial
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score


@dataclass
class Metric:
    name: str
    fn: Any


metrics = [
    Metric(
        name="precision",
        fn=lambda y_true, y_pred, y_pred_proba: accuracy_score(y_true, y_pred),
    ),
    Metric(
        name="accuracy",
        fn=lambda y_true, y_pred, y_pred_proba: precision_score(y_true, y_pred),
    ),
    Metric(
        name="roc_auc",
        fn=lambda y_true, y_pred, y_pred_proba: roc_auc_score(y_true, y_pred_proba),
    ),
]


def read_csv(csv_file: str) -> pd.DataFrame:
    read_fn = (
        pd.read_csv
        if (csv_file.lower().endswith(".csv") or csv_file.lower().endswith(".csv.gz"))
        else pd.read_parquet
    )

    # First find all the columns and just select which columns we don't need
    columns = read_fn(csv_file, nrows=1).columns
    columns = [str(c) for c in columns]
    columns = [c for c in columns if not c.startswith("exclude")]
    columns = [c for c in columns if not c.startswith("Unnamed:")]
    columns = [c for c in columns if c != "filename"]
    df = pd.read_csv(csv_file, usecols=columns)

    return df


def get_mean_std(df: pd.DataFrame) -> pd.Series:
    return (
        df.groupby("run_name").mean(),
        df.groupby("run_name").std(),
    )


def create_report(csv_file: str):
    df = read_csv(csv_file)

    # For every feaure set, run names start with 0 and end with 255. It
    # becomes easier to aggregate if run names were unique
    # Make them unique by simply appending the run name and feature set name
    df["run_name"] = df["feature_set"] + ":" + df["run_name"]

    def calculate_metrics(group):
        return pd.Series(
            {
                m.name: m.fn(group["y_true"], group["y_pred"], group["y_pred_proba"])
                for m in metrics
            }
        )

    metrics_df = df.groupby("run_name").apply(calculate_metrics)

    # At this point, the index is of the form fourier_only:run_56
    # Convert the index to a column
    metrics_df.reset_index(inplace=True)

    # At this point the run name is of the form fourier_only:run_52
    # What we now want to do is aggregate again with only the first part
    # of the run name, that is fourier_only. So we just modify run_name
    # and then do the rest
    metrics_df["run_name"] = metrics_df["run_name"].str.split(":").str[0]

    mean_df = pd.DataFrame()
    std_df = pd.DataFrame()
    aggregated_df = pd.DataFrame()
    for m in metrics:
        mean_series, std_series = get_mean_std(metrics_df[["run_name", m.name]])
        mean_df[m.name] = mean_series
        std_df[m.name] = std_series
        aggregated_df[f"{m.name}-mean"] = mean_series
        aggregated_df[f"{m.name}-std"] = std_series

    return metrics_df, mean_df, std_df, aggregated_df


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="GenReport", description="From a combined dataframe, create a report"
    )
    parser.add_argument("-f", "--file", type=str, required=True)
    args = parser.parse_args()
    create_report(args.file)


if "__main__" == __name__:
    main()
