#!/bin/bash
# GPU validation run: the test suite on the A100, then the cost profile (it saves after every cell). Submit from the repository root:
#   sbatch -A <account> [-p plgrid-now] scheduler/slurm_validate.sh
#SBATCH --job-name=hcr2-validate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=00:58:00
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -o pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
module purge
module load Python/3.10.4 CUDA/12.4.0
source venv/bin/activate || exit 1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv,noheader
git rev-parse HEAD
srun python tests2/run_tests2.py pre --device cuda
srun python tests2/run_tests2.py network --device cuda
srun python scheduler/profile_cost.py --device cuda
