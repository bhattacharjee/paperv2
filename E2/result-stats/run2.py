#!/usr/bin/env python3

from pathlib import Path
import os
import tqdm

BASE_DIR = "/Users/phantom/dev/paperv2/E2/"
PROG = str(Path(BASE_DIR, "summary_report.py"))

for dirname in tqdm.tqdm(["all", "password", "office"]):
    for metric in tqdm.tqdm(["auc_roc", "auc_prc"]):
        thedirectory = str(Path(BASE_DIR, "result-stats", dirname))
        output_file = f"summary{os.path.sep}summary.{dirname}.{metric}.csv"
        command = f"python3 {PROG} -d {thedirectory} -o {output_file} -m {metric}"
        os.system(command)
