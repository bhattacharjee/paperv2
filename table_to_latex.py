import argparse
from dataclasses import dataclass
from typing import Final, List

import pandas as pd

ROUND_DECIMALS: Final = 3


@dataclass
class Feature:
    name: str
    order: int


feature_name_map = {
    "baseline-only": Feature("Baseline only", 0),
    "advanced-only": Feature("Advanced only", 1),
    "fourier-min-only": Feature("Fourier (min) only", 2),
    "fourier-only": Feature("Fourier only", 3),
    "baseline-and-advanced": Feature("Baseline and advanced", 4),
    "baseline-and-fourier-min": Feature("Baseline and Fourier (min)", 5),
    "baseline-and-fourier": Feature("Baseline and Fourier", 6),
    "advanced-and-fourier-min": Feature("Advanced and Fourier (min)", 7),
    "advanced-and-fourier": Feature("Advanced and Fourier", 8),
    "baseline-advanced-and-fourier-min": Feature(
        "Baseline, advanced and Fourier (min)", 9
    ),
    "baseline-advanced-and-fourier": Feature("Baseline, advanced and Fourier", 10),
}

# The order of columns used is the order in this map
rename_columns_map = {
    "run_name": "Feature Set",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-Score",
    "roc_auc": "ROC\\_AUC",
    "auc_pr": "PRC\\_AUC",
    "tpr": "TPR",
    "tnr": "TNR",
    "fpr": "FPR",
    "fnr": "FNR",
}


def highlight_extremes(
    df: pd.DataFrame,
    columns: List[str],
    descending: bool = False,
    round_decimals: int = ROUND_DECIMALS,
) -> pd.DataFrame:
    df = df.copy()
    df = df[[c for c in rename_columns_map]]  # reorder

    close_bracket = "}"
    min_marker = "\\textOrange{"
    max_marker = "\\textBlue{"
    if descending:
        min_marker, max_marker = max_marker, min_marker

    def myround(x, ndigits):
        x = round(x, ndigits)
        x = str(x)
        while len(x) <= ndigits + 1:
            x = x + "0"
        return x

    for column in columns:
        min_value = df[column].min()
        max_value = df[column].max()

        def transform_cell(x):
            lmax_marker, lmin_marker = max_marker, min_marker

            # for fnr, fpr, the highlighting is reversed (but not for standard
            # deviation - which is where descending=True is specified)
            if not descending and column in ["fnr", "fpr", "FPR", "FNR"]:
                lmin_marker, lmax_marker = lmax_marker, lmin_marker

            if x == min_value:
                return f"{lmin_marker}{myround(x, round_decimals)}{close_bracket}"
            elif x == max_value:
                return f"{lmax_marker}{myround(x, round_decimals)}{close_bracket}"
            else:
                return str(myround(x, round_decimals))

        df[column] = df[column].apply(transform_cell)

    columns_to_drop = list(set(df.columns) - set(columns))
    columns_to_drop = [c for c in columns_to_drop if "name" not in c.lower()]
    df.drop(columns=columns_to_drop, inplace=True)
    print(f"{columns_to_drop=}")
    return df


def transform_run_names(df: pd.DataFrame) -> pd.DataFrame:
    """Transform the run names"""
    df = df.copy()
    order_map = {name: v.order for name, v in feature_name_map.items()}
    name_map = {name: v.name for name, v in feature_name_map.items()}
    df["sort_order"] = df["run_name"].map(lambda x: order_map[x])
    df["run_name"] = df["run_name"].map(lambda x: name_map[x])
    df = df.sort_values(by="sort_order")
    df.drop(columns=["sort_order"])
    return df


def main():
    parser = argparse.ArgumentParser("Convert CSV to json")
    parser.add_argument("-f", "--file", type=str, required=True, help="Input file")
    parser.add_argument(
        "-r",
        "--round-decimals",
        type=int,
        default=ROUND_DECIMALS,
        help="Number of decimal points.",
    )
    parser.add_argument(
        "-d",
        "--descending",
        type=bool,
        default=False,
        action="store_true",
        help="Descending order?",
    )
    args = parser.parse_args()
    filename = args.file

    if (
        filename.lower().endswith(".parquet")
        or filename.lower().endswith(".parquet.gz")
        or filename.lower().endswith(".pq")
        or filename.lower().endswith(".pq.gz")
    ):
        df = pd.read_parquet(filename)
    else:
        df = pd.read_csv(filename)

    df = transform_run_names(df)

    h_df = highlight_extremes(
        df,
        columns=[
            "precision",
            "recall",
            "roc_auc",
            "auc_pr",
            "accuracy",
            "f1",
            "fnr",
            "fpr",
            "tnr",
            "tpr",
        ],
        round_decimals=args.round_decimals,
        descending=args.descending,
    )
    h_df = h_df.rename(columns=rename_columns_map)
    latex_table = h_df.to_latex(index=False, escape=False)

    print(latex_table)


if __name__ == "__main__":
    main()
