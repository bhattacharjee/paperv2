import argparse
import os
import re
from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class Metric:
    name: str
    fn: Any


def get_confusion_matrix_unraveled(
    y_true: pd.Series, y_pred: pd.Series, which_rate: str
):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (tn + fp)
    fnr = fn / (tp + fn)

    columns = ["tnr", "fpr", "fnr", "tpr"]
    output = [tnr, fpr, fnr, tpr]
    return output[columns.index(which_rate)]


fpr = partial(get_confusion_matrix_unraveled, which_rate="fpr")
fnr = partial(get_confusion_matrix_unraveled, which_rate="fnr")
tpr = partial(get_confusion_matrix_unraveled, which_rate="tpr")
tnr = partial(get_confusion_matrix_unraveled, which_rate="tnr")


def auc_pr(y_true: pd.Series, y_pred_proba: pd.Series) -> np.float64:
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)


metrics = [
    Metric(
        name="precision",
        fn=lambda y_true, y_pred, y_pred_proba: precision_score(
            y_true, y_pred, zero_division=0
        ),
    ),
    Metric(
        name="recall",
        fn=lambda y_true, y_pred, y_pred_proba: recall_score(
            y_true, y_pred, zero_division=0
        ),
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
        fn=lambda y_true, y_pred, y_pred_proba: f1_score(
            y_true, y_pred, zero_division=0
        ),
    ),
    Metric(
        name="roc_auc",
        fn=lambda y_true, y_pred, y_pred_proba: roc_auc_score(y_true, y_pred_proba),
    ),
    Metric(
        name="auc_pr",
        fn=lambda y_true, y_pred, y_pred_proba: auc_pr(y_true, y_pred_proba),
    ),
    Metric(name="tpr", fn=lambda y_true, y_pred, y_pred_proba: tpr(y_true, y_pred)),
    Metric(name="tnr", fn=lambda y_true, y_pred, y_pred_proba: tnr(y_true, y_pred)),
    Metric(name="fpr", fn=lambda y_true, y_pred, y_pred_proba: fpr(y_true, y_pred)),
    Metric(name="fnr", fn=lambda y_true, y_pred, y_pred_proba: fnr(y_true, y_pred)),
]


def csv_read_fn(csv_file: str) -> Any:
    read_fn = (
        pd.read_csv
        if (csv_file.lower().endswith(".csv") or csv_file.lower().endswith(".csv.gz"))
        else pd.read_parquet
    )
    return read_fn


def is_interesting_file(x):
    """We don't want files that are the PDF descriptions
    of each folder"""
    pattern = r".*\d{4}\-.*"
    m = re.match(pattern, x)
    if m:
        return True
    return False


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

    # TODO: Needs some cleaning up.
    # 1. Find a better way to deal with read_parquet and read_csv
    # 2. In read_pandas, find a way to read only the first row

    # First find all the columns and just select which columns we don't need
    if read_fn == pd.read_csv:
        columns = read_fn(csv_file, nrows=1).columns
    else:
        columns = read_fn(csv_file).columns
    columns = [str(c) for c in columns]
    columns = [c for c in columns if not c.startswith("exclude")]
    columns = [c for c in columns if not c.startswith("Unnamed:")]
    if read_fn == pd.read_csv:
        df = read_fn(csv_file, usecols=columns)
    else:
        df = read_fn(csv_file, columns=columns)

    df = df[df["filename"].map(is_interesting_file)]

    return df


def get_mean_std(df: pd.DataFrame) -> pd.Series:
    return (
        df.groupby("run_name").mean(),
        df.groupby("run_name").std(),
    )


def create_report_for_df(
    df: pd.DataFrame,
    matches: Optional[Union[List[str], str]] = None,
    negative_matches: Optional[Union[List[str], str]] = None,
):
    df = df.copy()

    def is_row_match(x):
        if not negative_matches and not matches:
            return True
        if negative_matches:
            if isinstance(negative_matches, list):
                for x1 in negative_matches:
                    if x1.lower() in x:
                        return False
            else:
                if negative_matches.lower() in x.lower():
                    return False
        if matches:
            if isinstance(matches, list):
                for x1 in matches:
                    if x1.lower() in x.lower():
                        return True
            else:
                if matches.lower() in x.lower():
                    return True
        return False

    df = df[df["filename"].map(is_row_match)]
    df.drop(columns=["filename"], axis=1, inplace=True)

    def calculate_metrics(group):
        return pd.Series(
            {
                m.name: m.fn(group["y_true"], group["y_pred"], group["y_pred_proba"])
                for m in metrics
            }
        )

    metrics_df = df.groupby("feature_set").apply(calculate_metrics)
    metrics_df.reset_index(inplace=True)

    return metrics_df


def create_report(input_csv_file: str, output_csv_file: str):
    # read_fn = csv_read_fn(input_csv_file)
    df = read_csv(input_csv_file)

    # def is_password_protected(filename):
    #     return "password" in filename.lower()

    # def is_archive(filename):
    #     for ext in [".tar.gz", ".gzip", ".zip", ".tar", ".7z"]:
    #         if filename.lower().endswith(ext):
    #             return True
    #     return False

    # def is_office_file(filename):
    #     for ext in [".ppt", ".pptx", ".xls", ".xlsx", ".doc", ".docx", ".odp", ".odt", ".ods"]:
    #         if filename.lower().endswith(ext):
    #             return True
    #     return False

    password_result = create_report_for_df(df, ["password"], [".7z", ".gz", ".zip"])
    all_files_result = create_report_for_df(df)
    office_result = create_report_for_df(
        df, [".xls", ".csv", ".ppt", ".doc", ".doc", "odf", "opf"]
    )

    print("*************** ALL FILES ****************")
    print(all_files_result.sort_values(by="auc_pr"))
    print("*************** PASSWORD PROTECTED FILES ****************")
    print(password_result.sort_values(by="auc_pr"))
    print("*************** OFFICE FILES ****************")
    print(office_result.sort_values(by="auc_pr"))


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
    create_report(args.input_file, args.output_file)


if "__main__" == __name__:
    main()
