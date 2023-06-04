#!/usr/bin/env python3

import argparse
import gc
import glob
import json
import os
from functools import lru_cache
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import tqdm

from loguru import logger


def call_gc():
    for _ in range(3):
        for j in range(3):
            gc.collect(j)


def get_feature_sets(s: pd.Series) -> List[str]:
    return s.unique().tolist()


g_counter = 1


def get_counter() -> int:
    global g_counter
    g_counter += 1
    return g_counter


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
            run_name = lfilename.split(os.path.sep)[-2]
            base_filename = lfilename.split(os.path.sep)[-1]
            configuration = get_configuration_json(lfilename)
            df = pd.read_csv(lfilename)
            for colname, colval in configuration.items():
                df[colname] = colval
                df[colname] = df[colname].astype(np.bool_)
            df["y_pred"] = df["y_pred"].astype(np.int8)
            df["y_true"] = df["y_true"].astype(np.int8)
            df["feature_set"] = feature_set
            # df["feature_set"] = df["feature_set"].astype("categorical")
            df["run_name"] = run_name
            df["filename"] = base_filename
            dataframes.append(df)

    logger.info("Concatenating dataframes ...")
    df = pd.concat(dataframes)
    df = df[df["exclude_plaintext_nonbase32"] == False]
    del dataframes
    dataframes = None
    logger.info("OK.")

    call_gc()

    return df


def main(run_as_prog: bool = False) -> None:
    parser = argparse.ArgumentParser("Generate report from runs")
    parser.add_argument("-i", "--input-directory", type=str, required=True)
    parser.add_argument("-o", "--output-file-name", type=str, required=True)

    if run_as_prog:
        args = parser.parse_args()

    df = load_results(args.input_directory)
    logger.info("Saving output file...")
    df.to_csv(args.output_file_name)
    logger.info("OK.")


if __name__ == "__main__":
    logger.add(
        "./combine_dfs.log", backtrace=True, diagnose=True, level="INFO"
    )
    main(True)
