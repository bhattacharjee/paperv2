import argparse
import glob
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict

import pandas as pd
from loguru import logger


class Metric(Enum):
    AUC_ROC = "AUC_ROC"
    AUC_PRC = "AUC_PRC"
    F1_SCORE = "F1"
    ACCURACY = "ACCURACY"


@dataclass
class MetricsAndModels:
    filename: str
    modelname: str
    dataframe: Optional[pd.DataFrame] = None


def glob_filenames(dirname: str, use_stddev: bool = False) -> List[str]:
    out_filenames: List[str] = []
    extra_pattern = "std*" if use_stddev else "mean*"
    for exten in ("*.csv", "*.csv.gz", "*.pq", "*.pq.gz", "*.parquet", "*.parquet.gz"):
        out_filenames = out_filenames + glob.glob(f"{dirname}/{extra_pattern}{exten}")
    if not out_filenames:
        for exten in ("*.csv", "*.csv.gz", "*.pq", "*.pq.gz", "*.parquet", "*.parquet.gz"):
            out_filenames = out_filenames + glob.glob(f"{dirname}/{exten}")
    return out_filenames


def get_dataframes(dir_name: str, use_stddev: bool = False) -> List[MetricsAndModels]:
    out_list: List[MetricsAndModels] = []
    for fname in glob_filenames(dir_name, use_stddev):
        b_fname = os.path.basename(fname)
        if (
            "knn" in b_fname.lower()
            or "neighbour" in b_fname.lower()
            or "neighbor" in b_fname.lower()
        ):
            out_list.append(
                MetricsAndModels(filename=fname, modelname="k Nearest Neighbours")
            )
        elif "gaussian" in b_fname.lower():
            out_list.append(
                MetricsAndModels(filename=fname, modelname="Gaussian Naive Bayes")
            )
        elif "lda" in b_fname.lower() or (
            "linear" in b_fname.lower() and "discriminant" in b_fname.lower()
        ):
            out_list.append(MetricsAndModels(filename=fname, modelname="LDA"))
        elif (
            "lr" in b_fname.lower()
            or ("logistic" in b_fname.lower() and "regression" in b_fname.lower())
            or ("linear" in b_fname.lower() and "regression" in b_fname.lower())
        ):
            out_list.append(
                MetricsAndModels(filename=fname, modelname="Logistic Regression")
            )
        elif (
            "neural" in b_fname.lower()
            or "network" in b_fname.lower()
            or ".nn." in b_fname.lower()
            or "_nn_" in b_fname.lower()
        ):
            out_list.append(
                MetricsAndModels(filename=fname, modelname="Neural Network")
            )
        elif (
            "qda" in b_fname.lower()
            or "quadratic" in b_fname.lower()
            or "quadriatic" in b_fname.lower()
        ):
            out_list.append(MetricsAndModels(filename=fname, modelname="QDA"))
        elif (
            "random" in b_fname.lower()
            or "forest" in b_fname.lower()
            or ".rf." in b_fname.lower()
            or "_rf_" in b_fname.lower()
        ):
            out_list.append(MetricsAndModels(filename=fname, modelname="Random Forest"))

        else:
            logger.error(f"Could not determine type for {fname=}")

    for element in out_list:
        if ".csv" in element.filename.lower():
            element.dataframe = pd.read_csv(element.filename)
        elif (
            ".pq" in element.filename.lower() or ".parquet" in element.filename.lower()
        ):
            element.dataframe = pd.read_parquet(element.filename)
        else:
            logger.error(f"{element.filename=} has incorrect extension")
            raise Exception(f"{element.filename=} has incorrect extension")

    return out_list


def get_chosen_series(df: pd.DataFrame, metric: Metric) -> pd.Series:
    columns = [str(c) for c in df.columns]

    def rename_run_name(n: str) -> str:
        if n == "fourier-min":
            return "fourier-min-only"
        return n

    chosen_column = None
    if metric == Metric.AUC_PRC:
        for c in columns:
            if "prc" in c.lower() or ("auc" in c.lower() and "pr" in c.lower()):
                chosen_column = c
                break
    elif metric == Metric.AUC_ROC:
        for c in columns:
            if "auroc" in c.lower() or ("auc" in c.lower() and "roc" in c.lower()):
                chosen_column = c
                break
    elif metric == Metric.ACCURACY:
        for c in columns:
            if "acc" in c.lower() or "accuracy" in c.lower():
                chosen_column = c
                break
    elif metric == Metric.F1_SCORE:
        for c in columns:
            if "f1" in c.lower():
                chosen_column = c
                break
    else:
        raise Exception(f"Invalid {metric=}")

    if not chosen_column:
        raise Exception(f"Could not find column for {metric=}")
    
    df = df.reset_index(drop=True)

    if "run_name" in df.columns:
        df["run_name"] = df["run_name"].map(rename_run_name)
        df = df.set_index("run_name")
    elif "feature_set" in df.columns:
        df["feature_set"] = df["feature_set"].map(rename_run_name)
        df["run_name"] = df["feature_set"]
        df = df.set_index("run_name")
    return df[chosen_column]


def create_summary(mam: List[MetricsAndModels], metric: Metric) -> pd.DataFrame:
    df = pd.DataFrame()
    for e in mam:
        df[e.modelname] = get_chosen_series(e.dataframe, metric)

    # Sort the columns so that we have the same order
    columns = sorted([str(c) for c in df.columns])
    df = df[columns]
    
    def rename_run_name(n: str) -> str:
        if n == "fourier-min":
            return "fourier-min-only"
        return n

    if "run_name" in df.columns:
        df["run_name"] = df["run_name"].map(rename_run_name)
        df = df.set_index("run_name")
    elif "feature_set" in df.columns:
        df["run_name"] = df["feature_set"].map(rename_run_name)
        df.drop(["feature_set"], inplace=True)
        df.set_index("run_name", inplace=True)

    return df


@dataclass
class BiggestDifference:
    modelname: str
    improvement: str

def get_best_difference_between_two_rows(df: pd.DataFrame, row1: str, row2: str) -> Dict[str, BiggestDifference]:
    """
    First, compute row2 - row1, and then get the best value
    """
    difference = df.loc[row2] - df.loc[row1]
    models: List[str] = list()
    differences: List[float] = list()
    for m, d in difference.to_dict().items():
        models.append(m)
        differences.append(d)
    df = pd.DataFrame({"models": models, "differences": differences})
    max_difference = df["differences"].max()
    df = df[df["differences"] == max_difference]
    df = df.head(1).reset_index(drop=True)
    retval = {
        row2: BiggestDifference(
            modelname = df.loc[0]["models"],
            improvement = df.loc[0]["differences"]
        )
    }
    return retval

def get_best_values(df: pd.DataFrame) -> Dict[str, BiggestDifference]:
    retval = {}
    d = get_best_difference_between_two_rows(df, "baseline-only", "baseline-advanced-and-fourier")
    retval.update(d)
    d = get_best_difference_between_two_rows(df, "baseline-only", "baseline-advanced-and-fourier-min")
    retval.update(d)
    return retval

def main():
    parser = argparse.ArgumentParser(prog="Create summary report")
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=True,
        help="Input directory where all the files are located",
    )
    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["auc_roc", "roc_auc", "auroc", "auc_prc", "prc_auc", "f1", "accuracy"],
        required=True,
        help="Metric to choose from input files.",
    )
    parser.add_argument(
        "-o", "--output-file", type=str, required=False, help="Output file to write to"
    )
    parser.add_argument(
        "-s", "--stddev", action="store_true", help="Use the standard deviation instead of mean"
    )
    args = parser.parse_args()

    if args.metric in ("auc_roc", "roc_auc", "auroc"):
        metric = Metric.AUC_ROC
    elif args.metric in ("auc_prc", "prc_auc"):
        metric = Metric.AUC_PRC
    elif args.metric in ("f1"):
        metric = Metric.F1_SCORE
    elif args.metric in ("accuracy"):
        metric = Metric.ACCURACY
    else:
        raise Exception(f"Unexpected metric value {metric}")

    metrics_and_models = get_dataframes(args.directory, args.stddev)
    summary = create_summary(metrics_and_models, metric)
    best_values = get_best_values(summary)
    for name, value in best_values.items():
        print(f"{name:40.40s} | {value.modelname:30.30s} | {value.improvement:0.3f}")
    # if args.output_file:
    #     if args.output_file.lower().endswith(
    #         "csv"
    #     ) or args.output_file.lower().endswith(".csv.gz"):
    #         summary.to_csv(args.output_file)
    #     elif (
    #         args.output_file.lower().endswith("pq")
    #         or args.output_file.lower().endswith(".pq.gz")
    #         or args.output_file.lower().endswith(".parquet")
    #         or args.output_file.lower().endswith(".parquet.gz")
    #     ):
    #         summary.to_parquet(args.output_file)
    #     else:
    #         raise Exception(f"{args.output_file=} must be a parquet or csv")


if __name__ == "__main__":
    main()
