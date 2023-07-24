from launcher import KubernetesJob, WandbIdentifier, launch
import numpy as np
import random
from typing import List

def main(TASKS: list[str], job: KubernetesJob, name: str, group_name: str, synchronous=True):
    seed = 1259281515
    random.seed(seed)

    wandb_identifier = WandbIdentifier(
        run_name=f"{name}-{{i:05d}}",
        group_name=group_name,
        project=WANDB_PROJECT)

    commands: List[List[str]] = []
    for reset_network in RESET_NETWORK:
        for zero_ablation in ZERO_ABLATION:
            for task in TASKS:
                for metric in METRICS_FOR_TASK[task]:
                    # if "tracr" not in task:
                    #     if reset_network==0 and zero_ablation==0:
                    #         continue
                    #     if task in ["ioi", "induction"] and reset_network==0 and zero_ablation==1:
                    #         continue

                    command = [
                        "python",
                        "experiments/launch_param_integrated_grads.py",
                        f"--task={task}",
                        f"--wandb-run-name={wandb_identifier.run_name.format(i=len(commands))}",
                        f"--wandb-group={wandb_identifier.group_name}",
                        f"--wandb-entity=josephmiller101",
                        f"--wandb-project={wandb_identifier.project}",
                        f"--device=cpu",
                        f"--reset-network={reset_network}",
                        f"--seed={random.randint(0, 2**32 - 1)}",
                        f"--metric={metric}",
                        # f"--torch-num-threads={CPU}",
                        # "--wandb-dir=/root/.cache/huggingface/tracr-training/16heads",  # If it doesn't exist wandb will use /tmp
                        f"--wandb-mode=online",
                    ]
                    if zero_ablation:
                        command.append("--zero-ablation")

                    commands.append(command)

    launch(
        commands,
        name=wandb_identifier.run_name,
        job=job,
        check_wandb=wandb_identifier,
        just_print_commands=False,
        synchronous=synchronous,
    )

METRICS_FOR_TASK = {
    # "ioi": ["kl_div", "logit_diff"],
    # "tracr-reverse": ["l2"],
    # "tracr-proportion": ["kl_div", "l2"],
    # "induction": ["kl_div", "nll"],
    # "docstring": ["kl_div", "docstring_metric"],
    # "greaterthan": ["greaterthan", "kl_div"],
    "ioi": ["kl_div"],
    "tracr-reverse": ["l2"],
    "tracr-proportion": ["l2"],
    "induction": ["kl_div"],
    "docstring": ["kl_div"],
    "greaterthan": ["kl_div"]
}
# RESET_NETWORK = [0, 1]
RESET_NETWORK = [0]
# ZERO_ABLATION = [0, 1]
ZERO_ABLATION = [0]
HOFVARPNIR = False
WANDB_PROJECT = "pig-init-weights-fix-local-1"
TASKS = ["ioi", "greaterthan", "induction", "docstring", "tracr-reverse", "tracr-proportion"]
# TASKS = ["induction", "docstring", "tracr-reverse", "tracr-proportion"]
# CPU = 2

if __name__ == "__main__":
    if HOFVARPNIR:
        main(
            TASKS,
            KubernetesJob(container="ufo101/acdc-pig:1.0", cpu=2, gpu=0, memory="4Gi"),
            name="pig",
            group_name="pig",
            synchronous=False
        )
    else:
        main(
            TASKS,
            None,
            name="pig",
            group_name="pig",
            synchronous=False
        )
