"""Code of the study *Evidence or Marginal? What Moment-Vector Propagation Computes in Polynomial Density Neurons*.

The package is layered; imports point inwards only:

    domain     the computation: neurons, copulas, propagation, chains, job specifications, the pre-registered statistics.
               No files, no environment, no printing.
    ports      the interfaces through which use cases reach the outside: result store, job queue, record store, energy meter.
    app        the use cases: run a job, work through a job list, merge, analyse, build the paper's numbers, validate.
    adapters   the implementations of the ports (parquet, JSONL job lists with claim files, CSV/JSON records, NVML)
               and the output formats (matplotlib figures, LaTeX macros and tables, the arXiv archive).
    cli        the composition root: reads the environment, builds the adapters and runs a use case.
"""
