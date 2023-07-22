import argparse
import glob
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import pandas as pd
from loguru import logger


class Metric(Enum):
    AUC_ROC = "AUC_ROC"
    AUC_PRC = "AUC_PRC"


@dataclass
class MetricsAndModels:
    filename: str
    modelname: str
    dataframe: Optional[pd.DataFrame] = None


def glob_filenames(dirname: str, use_stddev: bool = False) -> List[str]:
    out_filenames: List[str] = []
    extra_pattern = ""
    for exten in ("*.csv", "*.csv.gz", "*.pq", "*.pq.gz", "*.parquet", "*.parquet.gz"):
        out_filenames = out_filenames + glob.glob(f"{dirname}/{extra_pattern}{exten}")
    return out_filenames


def get_dataframes(dir_name: str, use_stddev: bool = False) -> List[MetricsAndModels]:
    out_list: List[MetricsAndModels] = []
    for fname in glob_filenames(dir_name, use_stddev):
        print(f"{fname=}")
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

    def rename_feature_set(n: str) -> str:
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
    else:
        raise Exception(f"Invalid {metric=}")

    if not chosen_column:
        raise Exception(f"Could not find column for {metric=}")

    df["feature_set"] = df["feature_set"].map(rename_feature_set)
    df = df.set_index("feature_set")
    return df[chosen_column]


def create_summary(mam: List[MetricsAndModels], metric: Metric) -> pd.DataFrame:
    df = pd.DataFrame()
    for e in mam:
        df[e.modelname] = get_chosen_series(e.dataframe, metric)

    # Sort the columns so that we have the same order
    columns = sorted([str(c) for c in df.columns])
    df = df[columns]
    return df


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
        choices=["auc_roc", "roc_auc", "auroc", "auc_prc", "prc_auc"],
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
    else:
        raise Exception(f"Unexpected metric value {metric}")

    metrics_and_models = get_dataframes(args.directory, args.stddev)
    summary = create_summary(metrics_and_models, metric)
    print(summary)
    if args.output_file:
        if args.output_file.lower().endswith(
            "csv"
        ) or args.output_file.lower().endswith(".csv.gz"):
            summary.to_csv(args.output_file)
        elif (
            args.output_file.lower().endswith("pq")
            or args.output_file.lower().endswith(".pq.gz")
            or args.output_file.lower().endswith(".parquet")
            or args.output_file.lower().endswith(".parquet.gz")
        ):
            summary.to_parquet(args.output_file)
        else:
            raise Exception(f"{args.output_file=} must be a parquet or csv")


if __name__ == "__main__":
    main()
