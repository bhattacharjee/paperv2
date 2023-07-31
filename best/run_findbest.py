from dataclasses import dataclass
from typing import List, Dict, Tuple
import os
import subprocess
import sys

METRICS = ["prc_auc", "accuracy", "f1", "roc_auc"]

@dataclass
class RunDir:
    description: str
    directory: str
    
for metric in METRICS:
    filename = f"best_{metric}.txt"
    with open(filename, "w") as f:
        directories: List[RunDir] = [
            RunDir(description="E1", directory="E1/result-stats/summary-mean"),
            RunDir(description = "E2 - all files", directory = "E2/result-stats/all"),
            RunDir(description="E2 - password protected", directory = "E2/result-stats/password"),
            RunDir(description="E2 - office files", directory = "E2/result-stats/office"),
            RunDir(description="E3", directory = "E3/result-stats/mean")
        ]


        for rundir in directories:
            print(rundir.description, file=f)
            print("-" * 20, file=f)
            cmdline = f"python3 findbest.py -d {rundir.directory} --metric {metric}".split()
            try:
                process = subprocess.Popen(
                    args=cmdline,
                    stdout=subprocess.PIPE,
                    stderr=sys.stderr
                )
                output, _ = process.communicate()
                output = output.decode()
                process.wait()
                print(f"{output}\n\n", file=f)
            except Exception as e:
                print("EXCEPTION : ", e)
            print('-' * 80, file=f)
            print(file=f)
            print(file=f)