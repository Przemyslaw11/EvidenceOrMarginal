#!/bin/bash
# Site-specific settings (Athena, PLGrid): partition, modules and a venv/ at the repository root; adapt them to your cluster.
# Stage 7: one time/energy measurement run (gpu/energy.py) on one GPU, nothing else in the job.
# Submit from the repository root:
#   sbatch -A <account> -p plgrid-now --time=00:58:00 scheduler/slurm_energy.sh <float32|float64> <replicate>
#SBATCH --job-name=hcre
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=00:58:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -o pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
module purge
module load Python/3.10.4 CUDA/12.4.0
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}:${PYTHONPATH}"  # the package, without installing it
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.sm,power.limit,temperature.gpu --format=csv
nvidia-smi --query-compute-apps=pid,name --format=csv
git rev-parse HEAD
python -m gpu.energy --check && srun python -m gpu.energy --dtype "$1" --replicate "$2"
