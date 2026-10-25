# Pre-registration: scaled-up test of the constant-denominator approximation in HCR neurons

Status: frozen before any production job of Stages 3 and 4 (git history of this file is the record).
Deviations, if any, are added below under "Deviations" with a date and a reason; nothing above that line changes.

## 1. Question and approximation

An HCR neuron models the joint density of (x, y_1..y_k) in an orthonormal Legendre product basis. To propagate a
belief q(y), the paper divides by a constant instead of by the neuron's modelled input density D(y):
CONST_DENOM(x) = E_q[N(x, y)] / E_q[D(y)] (paper Eq. 10) instead of EXACT_UNCAL(x) = E_q[N(x, y) / D(y)].
This study measures the size of the difference, where D turns negative (after which EXACT_UNCAL is undefined), and how
both depend on the number of inputs, the basis, the ground truth, sampling noise, depth and training. It is an
independent numerical check of an approximation, not a criticism of HCR.

## 2. Outcome measures (exact definitions)

All TV distances are 0.5 int |p - r| on an 801-node grid in normal-score s on [-8, 8] (cubic sub-interval rule),
on raw (possibly signed, never renormalised) densities.

| name | definition |
|---|---|
| d_denom | TV(CONST_DENOM, EXACT_UNCAL), only where EXACT_UNCAL is defined |
| d_neg | TV(EXACT_UNCAL, EXACT_CAL), only where defined; EXACT_CAL: max(N, eps) normalised per input, eps in {0.1, 0.01, 0.001} |
| d_model_uncal, d_model_cal | TV(EXACT_UNCAL, GOLD), TV(EXACT_CAL, GOLD); GOLD = analytic conditional of the ground truth |
| d_total, d_dc | TV(CONST_DENOM, GOLD), TV(CONST_DENOM, EXACT_CAL) |
| relvar | (Var_CD - Var_ref) / Var_ref in x, ref = EXACT_UNCAL (primary) or GOLD |
| smean | abs(Mean_CD - Mean_ref) / sd_ref (standardised mean error) |
| minD | min of D over [0,1]^k: min(multi-start projected optimiser, MC minimum); an upper bound on the true minimum |
| P_Dneg_u | fraction of 2^20 uniform samples with D < 0 |
| chi_D | sqrt(int (D - 1)^2 dy) = sqrt(sum of squared a_(0, j), j != 0): distance of D from flat |
| defined | EXACT_UNCAL defined: no MC sample in supp(q) and no optimiser point in supp(q) with D < D_floor = 0.01 |

Primary settings: m = 4, eps = 0.01, beliefs S2 (flat), S3c (conflicting Beta), S4 (data: the true input copula).

**MC uncertainty.** R = 16 replicate blocks per belief; SE = sd(replicates)/sqrt(R); noise floor = mean
replicate-difference L1 / sqrt(R - 1). A value below 3 x its noise floor is flagged `below_noise` and is reported,
never subtracted. **Seed uncertainty.** Each seed re-estimates all coefficients; seed medians with 2.5-97.5 %
intervals; bootstrap over seeds (B = 2000, percentile 95 % CI, seed 12345), paired within seed when two conditions
are compared.

## 3. Thresholds

For a cell (experiment, type, k, basis, m, belief) and a metric X(rho_S):
- **Seed-median curve:** median over seeds at each grid point, with undefined seed values counted as +inf
  (conservative: undefined seeds are those with D < D_floor); defined at a grid point if >= 50 % of the seeds are defined there.
- **Threshold rho*_X(c):** the smallest rho_S at which the curve reaches c, by linear interpolation between adjacent
  grid points, searched over the run of defined points that starts at rho_S = 0. "Not reached" if the curve becomes
  undefined first or c is not reached by 0.9. Levels: d_denom >= 0.01 and >= 0.05; |relvar_vs_uncal| >= 0.05 and >= 0.20.
- **Negativity onset rho*_neg:** smallest rho_S at which the seed-median minD crosses 0 (linear interpolation);
  0 if minD < 0 already at rho_S = 0. **Undefined onset rho*_undef(b):** smallest grid rho_S at which >= 50 % of the
  seeds have EXACT_UNCAL undefined for belief b.
- Thresholds are also computed per seed; bootstrap CIs resample seeds and recompute the seed-median threshold
  ("not reached" counts as +inf in the percentile).

## 4. Grids (config2.yaml at the commit of this file)

- rho_S in {0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, ..., 0.5 step 0.05, 0.6, 0.7, 0.8, 0.9}; m in {2, 4, 6};
  eps in {0.1, 0.01, 0.001}; beliefs S1, S2, S3a-d, S4; M = 2^20 MC samples per belief; R = 16.
- **3a Gaussian** (`gauss`): k in {2, 4, 8, 16, 32}; types SUM, CONTRAST; bases pair, triple; n_est = 1e7;
  seeds 0-49 (S = 50; every cell of Q1 is treated as a headline cell).
- **3a population** (`population`): the same cells with exact Gaussian-copula coefficients (n_est = inf); 5 belief seeds.
- **3b families** (`families`): k in {2, 4, 8}; Gaussian-EXCH, t3 and t10 (SUM, CONTRAST, EXCH), Clayton-EXCH,
  Gumbel-EXCH, matched Kendall tau of the inputs; pair, triple; n_est = 1e7; seeds 0-19 (S = 20).
- **3b LKJ** (`lkj`): k in {2, 4, 8}; eta in {1, 4}; 200 random correlation matrices of (x, y) each; pair, triple; n_est = 1e7.
- **3c convergence** (`convergence`): k in {2, 8, 32}; SUM, CONTRAST; pair, triple; seeds 0-4; variants M x 2,
  n_est x 2, s-nodes 801 -> 1601, x-nodes 192 -> 384; paired with the same seeds of `gauss`.
- **Stage 4** (`network`): k in {2, 4, 8}; rho_S in {0, 0.2, 0.4, 0.6}; neuron-1 basis pair and triple; m = 4;
  L in {1, 2, 3, 4}; a = 0.6, b = 0.5; 10 seeds; beliefs S2, S3c, S4 (training) and S3a, S3b, S3d (held out);
  n_est = 1e6, n_train = 2e5, n_val = 1e5, n_test = 1e5.

**Execution order and budget rule.** Jobs run in the order gauss -> population -> convergence -> families -> lkj ->
network, within an experiment seed by seed (all cells of seed s before seed s + 1). The production budget is 300
GPU-h (target 255, 15 % reserve). If, after the first 10 GPU-h of production, the measured job wall times project the
total above 255 GPU-h, seeds are removed in this fixed order until the projection fits: gauss k = 32 triple to 30,
then 20 seeds; gauss k = 32 pair to 30, then 20; gauss k = 16 (both bases) to 30, then 20. No cell goes below 20
seeds (10 for `network`), no experiment is dropped, and only wall times enter this rule, never results. If the
projection still exceeds the budget, production stops before Stage 4 and the user decides. The analysed seeds of a
cell are seeds 0..S-1, all completed (failed jobs are listed); any change is recorded under Deviations.

## 5. Optimiser settings (fixed)

- min D: 2000 starts (corners, edge midpoints, 256 lowest-D MC samples of S2 and S4, uniform random), projected Adam,
  400 steps, cosine learning rate 0.005 -> 0.0001; projection onto supp(q) for the Beta beliefs. Chosen on the d = 3
  coefficients against brute-force grids (tests M9), not on production data.
- Stage 4 training (not tuned): Adam, learning rate 0.003 with cosine decay to 0, <= 2000 full-batch steps, validation
  every 50 steps, patience 10; L-BFGS (strong Wolfe), lr 1, 20 iterations per step, history 20, <= 50 steps,
  validation every step, patience 10. Best validation state kept. eps = 0.01 in the likelihood loss.
- Losses: T-val (Eq. 7 value MSE, rows i in {0, 1}; the all-rows variant is identical because its extra gradients are
  exactly zero, test M14d), T-nll (calibrated conditional log-likelihood, all rows), T-e2e (squared L2 distance between
  the CONST_DENOM output polynomial and the target density projected on the basis, summed over S2, S3c, S4; all rows).
  Initialisations: direct estimates, and the flat-D control (a_(0, j) = 0 for j != 0).

## 6. Hypotheses and decision rules

**Aggregation rule (used where a hypothesis is evaluated over cells).** In each evaluable cell the effect is
*significant* if its 95 % bootstrap CI excludes the no-effect value on the predicted side. A hypothesis is
**supported** if >= 2/3 of the evaluable cells are significant in the predicted direction; **not supported** if
>= 2/3 of the evaluable cells are not (CI includes no-effect or lies on the opposite side); **inconclusive**
otherwise, or with fewer than 3 evaluable cells. Structural hypotheses on population coefficients (no seed noise in
the coefficients) use the deterministic values: a cell counts as significant if the predicted inequality holds.

**Q1 (dimension)**
- **H1a** (structural): the negativity onset rho*_neg decreases with k. Cells: (type, basis, m = 4), population
  coefficients; significant if rho*_neg is strictly decreasing over k = 2, 4, 8, 16, 32.
- **H1b** (finite sample): the same with n_est = 1e7 (seed-median onsets, bootstrap CI of rho*_neg(32) - rho*_neg(2) < 0).
- **H1c** (collapse): where defined, d_denom under S2 is a single increasing function of chi_D, independent of k:
  for each (type, basis), a power law d_denom = C chi_D^beta fitted to all seed-median (k, rho_S) points (m = 4,
  population coefficients) has >= 90 % of the points within a factor 2 -> supported; < 50 % -> not supported;
  otherwise inconclusive. Points with chi_D = 0, or with d_denom flagged `below_noise` in >= 50 % of the seeds, are
  excluded (log of zero / noise-dominated).

**Q2 (basis)**
- **H2a**: higher degree gives an earlier negativity onset, rho*_neg(m = 6) < rho*_neg(4) < rho*_neg(2); cells (type,
  k, basis), population coefficients.
- **H2b**: the triple basis has an earlier negativity onset than the pair basis; cells (type, k >= 4, m = 4),
  population coefficients and, separately, n_est = 1e7.
- **H2c**: d_denom depends weakly on m: at fixed (type, k, basis, belief, rho_S) where all three m are defined,
  max/min over m of the seed-median d_denom <= 1.5 in >= 80 % of such points -> supported; <= 50 % -> not supported.
  Beliefs S2, S3c, S4; rho_S > 0; points where any m is flagged `below_noise` in >= 50 % of the seeds are excluded.

**Q3 (ground truth)**
- **H3a** (equivalence): the negativity onset of t3, t10, Clayton and Gumbel differs from the Gaussian of the same type
  by less than 0.05 in rho_S units (matched Kendall tau, m = 4, pair and triple, k in {2, 4, 8}). Supported if every
  bootstrap 95 % CI of the difference lies inside [-0.05, 0.05]; not supported (Gaussian-specific) if any CI lies
  entirely outside; inconclusive otherwise.
- **H3b**: d_denom under S2 at matched tau is within a factor 2 of the Gaussian (ratio CI inside [0.5, 2]); same rule,
  over all rho_S > 0 where both are defined and not `below_noise` (>= 50 % of seeds). Paired bootstrap where the two
  share random numbers (EXCH families vs Gaussian-EXCH), independent bootstraps otherwise (t-SUM/CONTRAST vs `gauss`).
- **H3c**: for LKJ matrices, the H1c power law (fitted on the equicorrelated Gaussian sweep of the same basis, type SUM)
  predicts d_denom (S2, m = 4) within a factor 2 for >= 80 % of the defined matrices -> supported; < 50 % -> not supported.

**Q4 (uncertainty)**
- **H4a**: estimation (seed) uncertainty dominates MC uncertainty: in >= 90 % of defined headline points
  (m = 4, eps = 0.01) the across-seed sd of d_denom exceeds the median within-seed MC SE -> supported; < 50 % -> not.
- **H4b** (= test M13): doubling M, n_est, the s-nodes or the x-nodes changes every reported TV by less than
  max(3 MC SE, 1e-3) on the 3c subset. Every violation is listed.

**Q5 (depth and training)**
- **H5a** (compounding): with direct estimates, d_total(L = 4) / d_total(L = 1) > 1. Cells (k, rho_S, basis, belief in
  S2, S3c, S4); seed-median ratio, CI above 1.
- **H5b**: training reduces d_denom of neuron 1 (S2, m = 4): paired change after - before < 0; cells (loss, optimiser,
  k, rho_S > 0, basis) with direct initialisation, evaluated where d_denom is defined before and after in >= 50 % of seeds
  (paired over those seeds). For the e2e loss the L = 4 run is used.
- **H5c**: training reduces P_Dneg_u of neuron 1: same rule, cells where the median P_Dneg_u before training > 0.
- **H5d**: end-to-end training reduces the network's d_total at layer L on the held-out beliefs S3a, S3b, S3d
  (per seed: mean over the three beliefs): paired change < 0; cells (optimiser, L, k, rho_S, basis).
- The flat-D control is reported alongside every training outcome as the "no-effect" reference; it does not enter
  the decision rules.

## 7. Reporting rules

All runs are reported, including failed jobs (listed) and undefined cells (shown as gaps or hollow markers).
Pilot data (seed index 1000) are excluded from every reported number. Every number in report2.md is generated by code
from the merged results. Optional Stage 7 (time/energy) is separate and never enters the error tables.

## Deviations

(none)
