#!/usr/bin/env python3

import argparse
import gc
import glob
import json
import os
from functools import lru_cache
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    plot_roc_curve,
    roc_curve,
    ConfusionMatrixDisplay,
)

color_names: List[Tuple[str, str]] = [
    ("baseline-only", "blue"),
    ("advanced-only", "cyan"),
    ("fourier-only", "magenta"),
    ("baseline-and-advanced", "green"),
    ("baseline-and-fourier", "darkorange"),
    ("advanced-and-fourier", "navy"),
    ("baseline-advanced-and-fourier", "turquoise"),
]

single_value_metrics: List[Tuple[str, str]] = [
    ("accuracy_score", accuracy_score),
    ("balanced_accuracy_score", balanced_accuracy_score),
    ("precision_score", precision_score),
    (
        "recall_score",
        recall_score,
    ),
    ("f1_score", f1_score),
    (
        "roc_auc_score",
        roc_auc_score,
    ),
]


def call_gc():
    for i in range(3):
        for j in range(3):
            gc.collect(j)


def get_feature_sets(s: pd.Series) -> List[str]:
    return s.unique().tolist()


def load_results(inp_directory: str) -> pd.DataFrame:
    dataframes = []

    @lru_cache(maxsize=128)
    def get_configuration_json(lfilename: str) -> Any:
        logfilename = os.path.dirname(lfilename) + os.path.sep + "log.log"
        with open(logfilename) as f:
            lines = [l.strip() for l in f.readlines()]
            saveline = None
            for line in lines:
                if (
                    "__main__:evaluate" in line
                    and "combination_json = " in line
                ):
                    saveline = line
                    break
            return json.loads(line.split("=")[1].strip())
        raise Exception("Could not find the configuration json")

    for ldirname in tqdm.tqdm(
        glob.glob(f"{inp_directory}{os.path.sep}*-*"), desc="Loading datasets"
    ):
        feature_set = ldirname.split(os.path.sep)[1]
        glob_pattern = f"{ldirname}{os.path.sep}*{os.path.sep}*.csv.gz"
        for lfilename in glob.glob(glob_pattern):
            configuration = get_configuration_json(lfilename)
            df = pd.read_csv(lfilename)
            for colname, colval in configuration.items():
                df[colname] = colval
                df[colname] = df[colname].astype(np.bool_)
            df["y_pred"] = df["y_pred"].astype(np.int8)
            df["y_true"] = df["y_true"].astype(np.int8)
            df["feature_set"] = feature_set
            dataframes.append(df)

    df = pd.concat(dataframes)
    df = df[df["exclude_plaintext_nonbase32"] == False]
    del dataframes
    dataframes = None

    call_gc()

    return df


def calculate_single_value_metrics(df: pd.DataFrame, feature_sets):
    dfdict = {}
    dfdict["metric_names"] = [s[0] for s in single_value_metrics]

    def get_metric_checked(fn, y_true, y_pred):
        try:
            return fn(y_true, y_pred)
        except Exception as e:
            return 1.1

    for feature in tqdm.tqdm(
        feature_sets, desc="Calculating single value metrics"
    ):
        tdf = df[df["feature_set"] == feature]
        dfdict[feature] = [
            get_metric_checked(fn, tdf["y_true"], tdf["y_pred"])
            for _, fn in single_value_metrics
        ]

    return pd.DataFrame(dfdict).T


def plot_roc(df: pd.DataFrame, feature_sets: List[str], outfile: str) -> None:
    for featureset, colorname in color_names:
        tdf = df[df["feature_set"] == featureset]
        fpr, tpr, thresholds = roc_curve(tdf["y_true"], tdf["y_pred"])
        plt.plot(fpr, tpr, label=featureset, color=colorname)
    plt.legend()
    plt.show()


def plot_confusion_matrix(
    df: pd.DataFrame, feature_sets: List[str], outfile: str
) -> None:
    n_features = len(feature_sets)
    # fig, ax = plt.subplots(1, n_features)
    for n, featureset in enumerate(feature_sets):
        tdf = df[df["feature_set"] == featureset]
        cm = confusion_matrix(
            df["y_true"].astype(np.bool_),
            (df["y_pred"] > 0.5).astype(np.bool_),
            labels=[True, False],
        )
        cf = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[True, False]
        )
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser("Generate report from runs")
    parser.add_argument("-i", "--input-directory", type=str, required=True)
    parser.add_argument(
        "-roc", "--plot-roc", action="store_true", default=False
    )
    parser.add_argument(
        "-cm", "--plot-confusion-matrix", action="store_true", default=False
    )
    args = parser.parse_args()

    df = load_results(args.input_directory)
    feature_sets = get_feature_sets(df["feature_set"])
    df = df[["y_true", "y_pred", "feature_set"]]
    single_value_metrics = calculate_single_value_metrics(df, feature_sets)
    single_value_metrics.to_csv(
        args.input_directory + os.path.sep + "metrics.csv"
    )

    if args.plot_roc:
        out_filename = args.input_directory + os.path.sep + "roc.png"
        plot_roc(df, feature_sets, out_filename)

    if args.plot_confusion_matrix:
        out_filename = args.input_directory + os.path.sep + "cm.png"
        plot_confusion_matrix(df, feature_sets, out_filename)
    print(single_value_metrics)


if __name__ == "__main__":
    main()
