# Evidence or Marginal?

Code, data and manuscript for *Evidence or Marginal? What Moment-Vector Propagation Computes in Polynomial Density Neurons* by Przemysław Spyra.

A neuron that stores a joint polynomial density, as in Hierarchical Correlation Reconstruction (HCR), can propagate a density over its inputs. The paper shows that the constant-denominator formula E[N]/E[D] and the exact form E[N/D] are not an approximation and its target. They are exact answers to different questions: Bayesian updating with the belief as virtual evidence, and Jeffrey's rule. It then measures how far apart they are and when each exists, in a pre-registered synthetic study with up to 32 inputs.

The manuscript is [`paper/main.pdf`](paper/main.pdf). The study protocol is [`PREREGISTRATION.md`](PREREGISTRATION.md).

## Layout

```
paper/        manuscript, figures, generated numbers and tables, post hoc analyses
gpu/          density neuron, copulas, propagation and energy benchmark (PyTorch)
network/      chains of neurons and training
analysis2/    thresholds, hypothesis tests, study figures and report
scheduler/    job lists, launcher and Slurm scripts
tests2/       validation tests M1-M14
results2/     merged results and summaries of the study
config2.yaml  study settings
src/, tests/, results/, config.yaml   the earlier three-input study
```

`results2/run_record.json` holds GPU accounting, and maps the commit ids stored in the run metadata to this repository's history.

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
sbatch -A <account> --array=0-39 scheduler/slurm_array.sh results2/jobs_gauss.jsonl
python scheduler/merge.py gauss population convergence families lkj network
python -m analysis2.thresholds gauss population families
python tests2/run_tests2.py post
python -m analysis2.hypotheses
python -m analysis2.figures
python -m analysis2.report
```

A job counts as done once its result file exists, so an array can be resubmitted until `merge.py` reports nothing missing.

The three-input study runs on one CPU core in a few minutes with `python run_all.py`.

## License

MIT, see [`LICENSE`](LICENSE).
