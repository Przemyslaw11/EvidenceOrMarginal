#!/bin/bash
# Site-specific settings (Athena, PLGrid): partition, modules, the home quota check and a venv/ at the repository root; adapt them to your cluster.
# One worker per array task (1 GPU each); all tasks pull from the same idempotent job list(s).
# Submit from the repository root:
#   sbatch -A <account> --array=0-39%40 --time=08:00:00 scheduler/slurm_array.sh results/jobs_gauss.jsonl [...]
#SBATCH --job-name=hcr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=08:00:00
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -o pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
module purge
module load Python/3.10.4 CUDA/12.4.0
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}:${PYTHONPATH}"  # the package, without installing it
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

USED_MB=$(quota -s | awk '/net\/people/ {getline; gsub(/M/,"",$1); print $1}')
if [ -n "$USED_MB" ] && [ "$USED_MB" -ge 9500 ]; then
    echo "HOME quota at ${USED_MB}M >= 9500M; refusing to run."
    exit 1
fi

nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv,noheader
git rev-parse HEAD
LEFT=$(squeue -h -j "$SLURM_JOB_ID" -o %L)
srun python scheduler/launcher.py --worker --device cuda ${LEFT:+--time-left "$LEFT"} --jobs "$@"
