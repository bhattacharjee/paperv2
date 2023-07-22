#!/usr/bin/env python3

import os
from loguru import logger
from pathlib import Path
import tqdm

BASE_DIRECTORY="/Users/phantom/dev/paperv2/E2/"

WHICH_SUBSET = "all"

command_options = {
    "all": "-a",
    "password": "-pwd",
    "office": "-off",
}

OUTPUT_BASE_DIRECTORY = str(Path(BASE_DIRECTORY, WHICH_SUBSET))

PYTHON_PROG = str(Path(BASE_DIRECTORY, "E2_report_from_combined.py"))

file_description = {
    "gaussian_naive_bayes": "gaussian_naive_bayes_combined_results.csv.gz",
    "kneighbours": "kneighbours_combined_results.csv.gz",
    "lda": "lda_combined_results.csv.gz",
    "logistic_regression": "logistic_regression_combined_results.csv.gz",
    "neural_networks": "neural_network_7layers_result.csv.gz",
    "qda": "qda_combined_results.csv.gz",
    "random_forest": "random_forest_combined_results.csv.gz",
}


for directory, option in tqdm.tqdm(command_options.items()):

    #logger.info(f"processing {directory=} {option=}")

    for description, filename in tqdm.tqdm(file_description.items()):
        input_filename = str(Path(BASE_DIRECTORY, "combined", filename))
        output_dirname = str(Path(BASE_DIRECTORY, "result-stats", directory))
        output_filename = str(Path(output_dirname, f"{description}.csv"))

        #logger.info(f"        processing {input_filename=} {output_dirname=}")

        if not os.path.exists(input_filename) or not os.path.exists(os.path.dirname(output_dirname)):
            raise Exception(f"{input_filename=} or {os.path.dirname(output_dirname)=} does not exist")

        command = f"python3 {PYTHON_PROG} -i {input_filename} -o {output_filename} {option}"

        os.system(command)
