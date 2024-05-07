import subprocess
import os
import time

# adapt for your own needs
SUBMISSION_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=gpt-j-6b_{train_size}_{random_state}
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --mem=40GB

conda activate chemlift

python -u submit_train_test_classif_photocatMOF.py {train_size} {random_state}"""


def submit(train_size, random_state):
    # create the submission script
    submission_script = SUBMISSION_TEMPLATE.format(
        train_size=train_size,
        random_state=random_state,
    )

    # write the submission script to a file, use settings in the filename
    filename = f"submit_{train_size}_{random_state}.sh"
    with open(filename, "w") as f:
        f.write(submission_script)

    # submit the job
    subprocess.run(["sbatch", filename])

    # remove the submission script
    os.remove(filename)


if __name__ == "__main__":
    for i in range(6):
        for train_size in [15, 25, 35, 45, 55, 65]:
            submit(train_size, i+42)
            time.sleep(1)