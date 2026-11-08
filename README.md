# Evidence or Marginal?

Code, data and manuscript for *Evidence or Marginal? What Moment-Vector Propagation Computes in Polynomial Density Neurons* by Przemysław Spyra.

A neuron that stores a joint polynomial density, as in Hierarchical Correlation Reconstruction (HCR), can propagate a density over its inputs. The paper shows that the constant-denominator formula E[N]/E[D] and the exact form E[N/D] are not an approximation and its target. They are exact answers to different questions: Bayesian updating with the belief as virtual evidence, and Jeffrey's rule. It then measures how far apart they are and when each exists, in a pre-registered synthetic study with up to 32 inputs.

The manuscript is [`paper/main.pdf`](paper/main.pdf). The study protocol is [`PREREGISTRATION.md`](PREREGISTRATION.md). It uses the file names of its time: `config2.yaml` is now `config.yaml`.

## Layout

```
paper/        manuscript, figures, generated numbers and tables, post hoc analyses
src/densprop/domain/  density neuron, copulas, propagation, chains and training (PyTorch); thresholds and hypothesis tests
gpu/          energy benchmark
network/      validation test of the chains (M14)
src/densprop/  job specs, sweeps, worker, merge, analysis, study figures, report and command line
scheduler/    cost model and Slurm scripts
tests/        validation tests M1-M14
results/      merged results and summaries of the study
config.yaml   study settings
three_input_study/   the earlier three-input study
```

`results/run_record.json` holds GPU accounting, and maps the commit ids stored in the run metadata to this repository's history. For each of those ids it also lists the git tree ids of the code that ran (`gpu`, `network`, `analysis2`, `scheduler`, `tests2`, `config2.yaml`, `PREREGISTRATION.md`, under the names of the time), so the map can be checked: `git rev-parse 8abbf12:gpu` gives the tree listed for `6a98646`.

## Requirements

Python 3.10 or later and the packages in `requirements.txt`; `requirements-lock.txt` pins the versions of the production runs. PyTorch is needed for the study (writing the job lists excepted), the tests, and `semantics.py`, `regression.py`, `check_theory.py` and `check_identity.py` in `paper/`; rebuilding the paper from the stored results does not need it.

## Rebuild the paper

Everything in the paper comes from the stored results. No GPU is needed.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cd paper
../.venv/bin/python make_figures.py
../.venv/bin/python make_numbers.py
latexmk -pdf main.tex
../.venv/bin/python check_numbers.py
```

`check_numbers.py` regenerates all numbers and tables and checks them against their sources. The per-run parquet files (`results/merged/<experiment>.parquet`, 517 MB) are not in the repository, so the checks that need them are skipped. With the six files placed in `results/merged/` (`gauss`, `population`, `convergence`, `families`, `lkj`, `network`), every check runs.

The full rebuild, from the repository root, regenerates the post hoc results and the study figures as well:

```bash
.venv/bin/python paper/posthoc.py        # needs the parquet files
.venv/bin/python paper/semantics.py      # PyTorch
.venv/bin/python paper/regression.py     # PyTorch
.venv/bin/python paper/check_theory.py   # needs the parquet files; PyTorch
.venv/bin/python -m densprop figures --paper --out paper/figures   # G1-G9; G8, G9 need the parquet files
cd paper
../.venv/bin/python make_figures.py
../.venv/bin/python make_numbers.py
latexmk -pdf main.tex
../.venv/bin/python check_numbers.py
../.venv/bin/python check_identity.py    # PyTorch
../.venv/bin/python make_arxiv.py
```

The outputs of the first four steps (`paper/*.json`) are in the repository, so the later steps run without them.

## Run the study

A reduced grid, under 30 minutes on one GPU:

```bash
python -m densprop small         # --device cpu also works, more slowly
```

The full study, on a Slurm cluster with PyTorch installed:

```bash
python -m densprop make-jobs gauss population convergence families lkj network
sbatch -A <account> --array=0-39 scheduler/slurm_array.sh results/jobs_gauss.jsonl
python -m densprop merge gauss population convergence families lkj network
python -m densprop thresholds gauss population families
python tests/run_tests.py post --out tests/test_results.json
python -m densprop hypotheses
python -m densprop figures
python -m densprop report
```

`sbatch` is repeated for each experiment's job list. A job counts as done once its result file exists, so an array can be resubmitted until the merge reports nothing missing. The Slurm scripts are site-specific: they load the cluster's Python and CUDA modules and expect a virtual environment in `venv/`.

The time and energy benchmark runs with `sbatch -A <account> scheduler/slurm_energy.sh <float32|float64> <replicate>`, once per setting and replicate; `python -m gpu.energy --summarise` then writes `results/energy/summary.csv` and `report.txt`.

The three-input study runs on one CPU core in a few minutes with `python three_input_study/run_all.py`.

## License

The code is licensed under Apache-2.0 ([`LICENSE`](LICENSE)). The manuscript, figures and data are licensed under CC BY 4.0 ([`LICENSE-CC-BY-4.0`](LICENSE-CC-BY-4.0)). See [`NOTICE`](NOTICE); to cite, use [`CITATION.cff`](CITATION.cff).
