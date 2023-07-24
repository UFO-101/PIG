#!/usr/bin/env python3
from experiments.launcher import KubernetesJob, launch
import numpy as np
import random
from typing import List
from pathlib import Path


# TASKS = ["ioi", "docstring", "greaterthan", "tracr-reverse", "tracr-proportion"]
TASKS = ["induction"]

# METRICS_FOR_TASK = {
#     "ioi": ["kl_div", "logit_diff"],
#     "tracr-reverse": ["kl_div"],
#     "tracr-proportion": ["kl_div", "l2"],
#     "induction": ["kl_div", "nll"],
#     "docstring": ["kl_div", "docstring_metric"],
#     "greaterthan": ["kl_div", "greaterthan"],
# }
METRICS_FOR_TASK = {
    "ioi": ["kl_div"],
    "tracr-reverse": ["l2"],
    "tracr-proportion": ["l2"],
    "induction": ["kl_div"],
    "docstring": ["kl_div"],
    "greaterthan": ["kl_div"],
}

# ALGS = ["pig", "16h", "sp", "acdc"]
ALGS = ["pig"]

# RESET_NETWORK = [0, 1]
RESET_NETWORK = [0]

# ZERO_ABLATION = [0, 1]
ZERO_ABLATION = [0]

repo_root = "/Users/josephmiller/Documents/Automatic-Circuit-Discovery"
OUT_DIR = Path(repo_root).resolve() / "experiments" / "results" / "init_weights_fix_plots_data"

def main():
    commands = []
    for alg in ALGS:
        for reset_network in RESET_NETWORK:
            for zero_ablation in ZERO_ABLATION:
                for task in TASKS:
                    for metric in METRICS_FOR_TASK[task]:
                        command = [
                            "python",
                            "notebooks/roc_plot_generator.py",
                            f"--task={task}",
                            f"--reset-network={reset_network}",
                            f"--metric={metric}",
                            f"--alg={alg}",
                            f"--out-dir={OUT_DIR}",
                        ]
                        if zero_ablation:
                            command.append("--zero-ablation")
                        commands.append(command)

    launch(commands, name="plots", job=None, synchronous=False)


if __name__ == "__main__":
    main()
