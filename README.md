# Evidence or Marginal?

Code, data and manuscript for *Evidence or Marginal? What Moment-Vector Propagation Computes in Polynomial Density Neurons* by Przemysław Spyra.

A neuron that stores a joint polynomial density, as in Hierarchical Correlation Reconstruction (HCR), can propagate a density over its inputs. The paper shows that the constant-denominator formula E[N]/E[D] and the exact form E[N/D] are not an approximation and its target. They are exact answers to different questions: Bayesian updating with the belief as virtual evidence, and Jeffrey's rule. It then measures how far apart they are and when each exists, in a pre-registered synthetic study with up to 32 inputs.

The manuscript is [`paper/main.pdf`](paper/main.pdf). The study protocol is [`PREREGISTRATION.md`](PREREGISTRATION.md). It uses the file names of its time: `config2.yaml` is now `config.yaml`.

## Layout

```
paper/        manuscript, figures, generated numbers and tables, post hoc analyses
gpu/          density neuron, copulas, propagation and energy benchmark (PyTorch)
network/      chains of neurons and training
analysis/     thresholds, hypothesis tests, study figures and report
scheduler/    job lists, launcher and Slurm scripts
tests/        validation tests M1-M14
results/      merged results and summaries of the study
config.yaml   study settings
three_input_study/   the earlier three-input study
```

`results/run_record.json` holds GPU accounting, and maps the commit ids stored in the run metadata to this repository's history.

## Requirements

Python 3.10 or later and the packages in `requirements.txt`; `requirements-lock.txt` pins the versions of the production runs. PyTorch is needed for the study (writing the job lists with `make_jobs.py` excepted), the tests, and `semantics.py`, `regression.py`, `check_theory.py` and `check_identity.py` in `paper/`; rebuilding the paper from the stored results does not need it.

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

`check_numbers.py` regenerates all numbers and tables and checks them against their sources. The per-run parquet files are not in the repository, so the checks that need them are skipped.

## Run the study

A reduced grid, under 30 minutes on one GPU:

```bash
python run_small.py              # --device cpu also works, more slowly
```

The full study, on a Slurm cluster with PyTorch installed:

```bash
python scheduler/make_jobs.py gauss population convergence families lkj network
sbatch -A <account> --array=0-39 scheduler/slurm_array.sh results/jobs_gauss.jsonl
python scheduler/merge.py gauss population convergence families lkj network
python -m analysis.thresholds gauss population families
python tests/run_tests.py post
python -m analysis.hypotheses
python -m analysis.figures
python -m analysis.report
```

A job counts as done once its result file exists, so an array can be resubmitted until `merge.py` reports nothing missing.

The three-input study runs on one CPU core in a few minutes with `python three_input_study/run_all.py`.

## License

MIT, see [`LICENSE`](LICENSE).
