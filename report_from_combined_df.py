import argparse
import os
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, f1_score, recall_score,
                             precision_recall_curve, precision_score,
                             roc_auc_score, balanced_accuracy_score)


@dataclass
class Metric:
    name: str
    fn: Any


@dataclass
class Result:
    metrics_df: pd.DataFrame
    mean_df: pd.DataFrame
    std_df: pd.DataFrame
    agg_df: pd.DataFrame


def auc_pr(y_true: pd.Series, y_pred_proba: pd.Series) -> np.float64:
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)


metrics = [
    Metric(
        name="precision",
        fn=lambda y_true, y_pred, y_pred_proba: precision_score(y_true, y_pred),
    ),
    Metric(
        name="recall",
        fn=lambda y_true, y_pred, y_pred_proba: recall_score(y_true, y_pred),
    ),
    Metric(
        name="accuracy",
        fn=lambda y_true, y_pred, y_pred_proba: accuracy_score(y_true, y_pred),
    ),
    Metric(
        name="balanced_accuracy",
        fn=lambda y_true, y_pred, y_pred_proba: balanced_accuracy_score(y_true, y_pred),
    ),
    Metric(
        name="f1",
        fn=lambda y_true, y_pred, y_pred_proba: f1_score(y_true, y_pred),
    ),
    Metric(
        name="roc_auc",
        fn=lambda y_true, y_pred, y_pred_proba: roc_auc_score(y_true, y_pred_proba),
    ),
    Metric(
        name="auc_pr",
        fn=lambda y_true, y_pred, y_pred_proba: auc_pr(y_true, y_pred_proba),
    ),
]

def csv_read_fn(csv_file: str) -> Any:
    read_fn = (
        pd.read_csv
        if (csv_file.lower().endswith(".csv") or csv_file.lower().endswith(".csv.gz"))
        else pd.read_parquet
    )
    return read_fn

def read_csv(csv_file: str) -> pd.DataFrame:
    parquet_filename = ""
    read_fn = csv_read_fn(csv_file)

    if csv_file.lower().endswith(".csv") or csv_file.lower().endswith(".csv.gz"):
        if csv_file.lower().endswith(".csv"):
            parquet_filename = f"{csv_file[:-4]}.parquet.gz"
        elif csv_file.lower().endswith(".csv.gz"):
            parquet_filename = f"{csv_file[:-7]}.parquet.gz"

    if parquet_filename and os.path.exists(parquet_filename):
        raise Exception(f"Parquet {parquet_filename} already exists. Remove and rerun.")

    # First find all the columns and just select which columns we don't need
    if read_fn == pd.read_csv:
        columns = read_fn(csv_file, nrows=1).columns
    else:
        columns = read_fn(csv_file).columns
    columns = [str(c) for c in columns]
    columns = [c for c in columns if not c.startswith("exclude")]
    columns = [c for c in columns if not c.startswith("Unnamed:")]
    columns = [c for c in columns if c != "filename"]
    if read_fn == pd.read_csv:
        df = read_fn(csv_file, usecols=columns)
    else:
        df = read_fn(csv_file, columns=columns)

    if parquet_filename:
        df.to_parquet(parquet_filename)

    return df


def get_mean_std(df: pd.DataFrame) -> pd.Series:
    return (
        df.groupby("run_name").mean(),
        df.groupby("run_name").std(),
    )


def create_report(input_csv_file: str, output_csv_file: str) -> Result:
    # read_fn = csv_read_fn(input_csv_file)
    df = read_csv(input_csv_file)

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

    return Result(
        metrics_df=metrics_df, mean_df=mean_df, std_df=std_df, agg_df=aggregated_df
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="GenReport", description="From a combined dataframe, create a report"
    )
    parser.add_argument("-i", "--input-file", type=str, required=True)
    parser.add_argument("-o", "--output-file", type=str, required=True)
    args = parser.parse_args()

    output_filename = (
        args.output_file
        if (
            args.output_file.lower().endswith(".csv")
            or args.output_file.lower().endswith(".csv.gz")
        )
        else "result_output.csv"
    )
    result = create_report(args.input_file, args.output_file)
    result.metrics_df.to_csv(f"metrics-{output_filename}")
    result.mean_df.to_csv(f"mean-{output_filename}")
    result.std_df.to_csv(f"std-{output_filename}")
    result.agg_df.to_csv(f"aggregate-{output_filename}")

    print(result.mean_df.sort_values(by="auc_pr"))
    print(result.std_df.sort_values(by="auc_pr"))
    return result


if "__main__" == __name__:
    main()
