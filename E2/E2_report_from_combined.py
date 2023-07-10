import argparse
import os
from dataclasses import dataclass
from functools import partial
from typing import Any
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, balanced_accuracy_score,
                             confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)


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
            y_true, y_pred, zero_division=1
        ),
    ),
    Metric(
        name="recall",
        fn=lambda y_true, y_pred, y_pred_proba: recall_score(
            y_true, y_pred, zero_division=1
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
            y_true, y_pred, zero_division=1
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


def get_order_number(name: str) -> int:
    order_of_columns = [
        "baseline-only",
        "advanced-only",
        "fourier-only",
        "fourier-min-only",
        "baseline-and-advanced",
        "baseline-and-fourier",
        "baseline-and-fourier-min",
        "advanced-and-fourier",
        "advanced-and-fourier-min",
        "baseline-advanced-and-fourier",
        "baseline-advanced-and-fourier-min",
    ]

    order_of_columns = {name1: order for order, name1 in enumerate(order_of_columns)}

    # Can be an alias in some runs
    order_of_columns["fourier-min"] = order_of_columns["fourier-min-only"]

    if name in order_of_columns:
        return order_of_columns[name]
    return -1


def get_combined_stats(df: pd.DataFrame) -> pd.DataFrame:
    def calculate_metrics(group):
        return pd.Series(
            {
                m.name: m.fn(group["y_true"], group["y_pred"], group["y_pred_proba"])
                for m in metrics
            })
        
    df = df.groupby("feature_set").apply(calculate_metrics).reset_index()
    df["order"] = df["feature_set"].map(get_order_number)
    df = df.sort_values(by="order", ignore_index=True)
    df.drop("order", axis=1, inplace=True)
    return df


def remove_base32_from_filename(fname: str) -> str:
    if fname.lower().startswith("base32.") or fname.lower().startswith("base64."):
        return fname[7:]
    return fname
    

def is_file_interesting(fname: str) -> bool:
    # There are some pdf files in each folder for description. those
    # must be removed. Non pdf files start with four digits
    return True if re.match(r'^\d{4}', fname) else False

def get_metrics_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    return get_combined_stats(df)


def is_password_protected_file(fname: str) -> bool:
    return "password" in fname.lower()

def is_office_file(fname: str) -> bool:
    extensions = [
        "csv",
        "pptx",
        "ics",
        "doc",
        "docx",
        "xlsx",
        "ppt",
        "txt",
        "xls",
        "ods",
        "oxps",
    ]
    for e in extensions:
        if fname.lower().endswith(e):
            return True
    return False


def print_latex(
    df: pd.DataFrame, highlight_min_max: bool = False, num_decimals: int = 3
) -> None:
    def format_min_max(
        x,
        min_value: float,
        max_value: float,
        num_decimals: int = 3,
        invert: bool = False,
    ) -> str:
        if isinstance(x, str):
            return x
        # if round(x, num_decimals) == round(min_value, num_decimals):
        if x == min_value:
            if not invert:
                return f"\\textOrange{{{x:.{num_decimals}f}}}"
            else:
                return f"\\textBlue{{{x:.{num_decimals}f}}}"
        # elif round(x, num_decimals) == round(max_value, num_decimals):
        elif x == max_value:
            if not invert:
                return f"\\textBlue{{{x:.{num_decimals}f}}}"
            else:
                return f"\\textOrange{{{x:.{num_decimals}f}}}"
        else:
            return f"{x:.{num_decimals}f}"

    def should_invert(colnames: str):
        invert = False
        if isinstance(colnames, tuple):
            if "std" in colnames:
                invert = True
            if colnames[0] in ["FPR", "FNR"]:
                invert = not invert
        return invert

    formatters = {
        colname: partial(
            format_min_max,
            min_value=df[colname].min(),
            max_value=df[colname].max(),
            num_decimals=num_decimals,
            invert=should_invert(colname),
        )
        for colname in df.columns
        if colname != ("feature_set", "")
    }

    # Columns are in two levels, each column is a tuple of two values
    # Get rid of some columns that we don't want to print
    df = df[
        [
            c
            for c in df.columns
            if (not isinstance(c, tuple) or c[0] not in ["Balanced-Accuracy"])
        ]
    ]
    df = df[
        [
            c
            for c in df.columns
            if (
                not isinstance(c, tuple)
                or not (c[0] in ["Accuracy"] and c[1] in ["std"])
            )
        ]
    ]
    df = df[
        [c for c in df.columns if c not in [("Precision", "std"), ("Recall", "std")]]
    ]
    df = df[
        [
            c
            for c in df.columns
            if not (c[0] in ["FPR", "TPR", "FNR", "TNR"] and c[1] == "std")
        ]
    ]

    print()
    if highlight_min_max:
        df_str = df.to_latex(formatters=formatters, escape=False, index_names=False)
    else:
        df_str = df.to_latex(index_names=False)
    print(df_str)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    
    # Remove the base32 from filename
    df['filename'] = df['filename'].map(remove_base32_from_filename)
    
    df = df[df['filename'].map(is_file_interesting)]
    # tdf = tdf[tdf['filename'] == 'quality-50-percent.pdf']
    # tdf = tdf[tdf['feature_set'] == 'baseline-only']
    # print(tdf)

    all_files: pd.DataFrame = get_metrics_comparisons(df.copy())

    print("ALL FILES")
    print("-" * len("ALL FILES"))
    print(all_files.round(3).reset_index(drop=True))
    # if args.to_latex:
    #     print_latex(
    #         all_files.reset_index(drop=True),
    #         args.highlight_min_max,
    #         args.num_decimals,
    #     )
    
    pwd_files: pd.DataFrame = get_metrics_comparisons(df[df['filename'].map(is_password_protected_file)])
    print()
    print("PASSWORD PROTECTED FILES")
    print("-" * len("PASSWORD PROTECTED FILES"))
    print(pwd_files.round(3).reset_index(drop=True))

    office_files: pd.DataFrame = get_metrics_comparisons(df[df['filename'].map(is_office_file)])
    print()
    print("OFFICE FILES")
    print("-" * len("OFFICE FILES"))
    print(office_files.round(3).reset_index(drop=True))



if "__main__" == __name__:
    main()
