#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=32gb
#SBATCH --job-name=snapshotensembles_a100_v2
#SBATCH --gres=gpu:1

module load devel/python/3.10.0_intel_19.1
/pfs/data5/software_uc2/bwhpc/common/devel/python/3.10.0_intel_19.1/python_intel_packages/bin/python -m pip install --upgrade pip

export PYKEEN_HOME="${TMP}/ashaban_ma"

pip install --requirement requirements.txt
python main.py
