# Motor Loss Reserving: Classical and Deep Learning Baselines

[![CI](https://github.com/Azeem-Ashmal/motor-loss-reserving-dl/actions/workflows/ci.yml/badge.svg)](https://github.com/Azeem-Ashmal/motor-loss-reserving-dl/actions/workflows/ci.yml)

Quarterly loss reserving on Malaysian comprehensive private car claims (Theft and
Windscreen perils), comparing classical actuarial methods (Mack Paid Chain
Ladder, Incurred Chain Ladder, Munich Chain Ladder, Bornhuetter-Ferguson)
against non-neural (Gradient Boosting) and neural (DeepTriangle-inspired LSTM)
machine learning baselines. This is the reserving pipeline behind an MSc
dissertation.

## Data availability

**No claims data is included in this repository.** The dissertation's results
were produced on a proprietary industry motor-claims database that cannot be
redistributed in any form or aggregation. `data/synthetic/` provides a generator that produces
a structurally identical, obviously-fake triangle set so the full pipeline can
be run and inspected by anyone. See `data/README.md` for the exact schema.

## Installation

```Shell
git clone <this repo>
cd <this repo>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python version and exact package versions used to produce the dissertation's
results are pinned in `requirements-lock.txt` and echoed in `env_block.tex`.

## Quickstart

Runs end to end on the bundled synthetic data (~2-5 minutes on CPU):

```Shell
python scripts/run_all.py
```

Writes `outputs/results.json` (every reserving/metric number the pipeline
computes) and, once implemented for a given figure, `outputs/figures/`.

## Reproducing the dissertation results

You need your own copy of equivalent quarterly claims triangles, laid out per
`data/README.md`. An optional `exposure_qtr.csv` per peril (same directory)
enables the Bornhuetter-Ferguson baseline; without it, BF is skipped and every
other model still runs.

```Shell
python scripts/run_all.py --data-dir /path/to/real/data --seed 42
python scripts/run_multiseed.py --peril theft --data-dir /path/to/real/data
python scripts/run_pooled_multiseed.py --data-dir /path/to/real/data
```

`run_all.py` produces the single-seed, single-peril reserving table for both
perils; `run_multiseed.py` produces the single-peril seed-variance distribution
(the explicit 20-seed list lives in `config/seeds.yaml`, not generated at
runtime, so the same seeds run every time); `run_pooled_multiseed.py` runs the
pooled multi-peril architecture (below) across the same 20 seeds.

### Pooled multi-peril architecture and interpretability

`src/pipeline.py`'s `run_pooled_lstm_evaluation` trains one LSTM on both
perils simultaneously, with a learned embedding telling it which peril each
cohort belongs to, rather than training two separate single-peril models.
This was proposed in the project's original research plan as a mitigation for
data scarcity on the thinner-trained peril, and is genuinely tested here
across a full 20-seed sweep (`scripts/run_pooled_multiseed.py`), not just
implemented and checked on one favourable run: the honest result is that it
does not reliably improve the data-scarce peril (the one promising single
seed was among the best 3 of 20, not typical) while it does reliably worsen
the data-rich one, exactly as the underlying data-scarcity hypothesis would
predict for the half of the prediction that actually holds (see the
dissertation, Chapters 4-6, for the full comparison and the reasoning).

`src/interpretability.py`'s `occlusion_analysis` answers "what is the model
actually using" by ablating one input channel at a time (replacing its entire
observed history with a neutral value) and measuring the resulting change in
holdout accuracy - a substitute for SHAP, whose standard implementations do
not have first-class support for variable-length recurrent models. A channel
whose removal *improves* accuracy was being used in a way that was actively
hurting the forecast; a channel whose removal *worsens* accuracy was being
used productively. `tests/test_interpretability.py` checks the harness itself
against a zero-weight model, which cannot respond to any input and must
therefore report exactly zero importance for both channels.

`scripts/run_pooled_ensemble.py` loads a completed `run_pooled_multiseed.py`
sweep's 20 checkpoints, rolls each forward autoregressively, and averages
predictions cell-by-cell to test the natural next question after a seed-
unstable architecture: would simple ensembling have fixed it for free? It
exists so that this exact, dissertation-cited result is reproducible from
code in this repository rather than from an ad-hoc calculation that was run
once and only its output kept - the original computation was exactly that,
an unsaved one-off script, which this repository's own "every number is
traceable to code" standard does not tolerate for long.

## Verified findings (2026-08-29 reconciliation)

Running this exact pipeline against the real underlying triangles surfaced a
determinism bug in an earlier iteration (PyTorch's own weight initialisation
was never seeded) and a genuine mistake in a hand-rolled Mack standard-error
formula (caught by cross-checking against `chainladder` on the published RAA
benchmark triangle, where it under-counted by roughly 1.85x). Once both were
fixed, the single-peril DeepTriangle LSTM does not reproduce the dissertation's
earlier claim of matching classical Incurred Chain Ladder accuracy - a 20-seed
re-run puts its mean Theft error at +34.2% (std ±35.4pp), significantly worse
than ICL's +1.3% (one-sample t-test, t=4.16, p=0.0005), and every one of the
20 seeds individually underperforms ICL.

That finding led to a follow-up test, not just a negative conclusion: an
occlusion analysis showed the single-peril model was over-extrapolating from
too little training data (1,128 cells), not ignoring its inputs. Pooling both
perils into one network with a peril embedding - proposed for exactly this
failure mode in the project's original research plan, before either result was
seen - looked like a striking fix on one seed (Theft error +50.9% to +4.4%,
$R^2$ 0.65 to 0.93). A full 20-seed re-run of the pooled architecture, run
before that result was accepted, shows it is not: the pooled 20-seed mean
(+47.2%, ±35.9pp) is statistically indistinguishable from the single-peril
mean (t=1.16, p=0.25) and remains significantly worse than ICL (t=5.73,
p=0.00002) - the promising seed was one of the three best draws out of twenty,
not typical. Windscreen's half of the same hypothesis (pooling should not help
a peril that was never data-scarce) does hold up under the same 20-seed
scrutiny, consistently worse than single-peril. See the dissertation's
Chapters 4-8 for the full discussion. This is exactly the kind of result the
reconciliation tests below exist to surface either way - they check
identities against whatever the code actually outputs, not against a target
answer.

A third hypothesis was tested the same way: Kuo (2019)'s original DeepTriangle
used a GRU cell, not an LSTM, so `DeepTriangleLSTM` now accepts
`rnn_type="lstm"|"gru"` (`--rnn-type` on `run_multiseed.py`) to test the cell
type directly rather than assume LSTM was the right choice. A 20-seed GRU
re-run on real Theft data gives a mean error of +39.9% (std ±43.4pp) -
statistically indistinguishable from the LSTM's +34.2% (independent t-test,
t=0.45, p=0.65) and still significantly worse than ICL (one-sample t-test
against ICL's +1.3%, t=3.97, p=0.0008; only 1 of 20 GRU seeds even matched
ICL's absolute error). Cell type is not the bottleneck either - reinforcing
the occlusion-analysis diagnosis that the limiting factor is training-data
volume (1,128 cells) and feature poverty, not a fixable architecture pick.

A fourth hypothesis then had to be tested rather than assumed: if feature
poverty is the real constraint, does adding a genuinely new feature help?
`run_peril_evaluation`/`run_multiseed.py --use-counts` add incremental claim
counts as a third LSTM input channel (from `claims_count_qtr_triangle.csv`,
same cumulative-triangle convention as paid/outstanding; future rollout steps
carry the last-observed count forward rather than leaking true future counts -
see `predict_auto_regressive`'s docstring in `src/evaluate.py`). The honest
result is the opposite of what the diagnosis predicted: a 20-seed re-run on
real Theft data gives mean error +65.6% (std ±47.6pp), significantly *worse*
than the 2-feature LSTM (+34.2%, independent t-test t=2.37, p=0.023) and worse
than ICL (t=6.04, p=0.000008; 0 of 20 seeds beat ICL). With only 1,128 training
cells, adding a third noisy channel increased the model's capacity to overfit
faster than it added usable signal - a real, if unwelcome, finding about this
specific dataset's size, not a defect in the idea of feature-richer models in
general.

Separately, `MackPaidChainLadder.mack_windowed_bootstrap` closes a gap flagged
earlier: `mack_se_full_runoff()` only ever covered the standard textbook
quantity (full run-off to ultimate), never the specific calendar-quarter
holdout window this pipeline's reserve figures actually use. The new method
is a genuine Mack-model simulation - not a different model bolted on - that
perturbs each cell's link ratio by Mack's own parameter variance
(sigma_j^2/sum C_ij) and process variance (sigma_j^2 * C_ij) recursively, then
sums only the incremental cells inside the target window for cohorts that
already existed as of the cutoff (matching `run_peril_evaluation`'s own
population exactly - excluding this restriction was an early bug caught by a
reconciliation test before real numbers were trusted, since it silently
included brand-new accident quarters that only begin inside the holdout
window). A second, more subtle bug was caught the same way on a later review
pass: the link-ratio parameter draw was originally sampled independently per
accident-year cohort rather than once per development period and shared
across cohorts within a simulation, which incorrectly diversifies away the
cross-cohort correlation Mack's own aggregate formula has a covariance term
for specifically because every cohort's projection uses the *same* uncertain
f_j. Fixed to draw one f_j per simulated development period, applied to every
cohort in that draw. On real data (8,000 simulations, post-fix): Theft's
windowed Mack reserve carries a coefficient of variation of 33.2% (mean
RM 79,933k, 90% interval RM 37,139k-124,551k), Windscreen's is tighter at
14.7% (mean RM 74,997k, 90% interval RM 56,435k-93,265k) - both reconcile to
within 1% of the deterministic chain-ladder point estimate for the same
window, as the model's own unbiasedness predicts. The fix moved these numbers
only slightly (pre-fix CVs were 33.15%/14.67%), suggesting parameter
uncertainty is a modest share of the total for this triangle, but the
methodology is now correct regardless of how much it happened to matter here.
Worth noting for context: Theft's LSTM mean error (+34.2%) sits
inside classical Mack's own uncertainty envelope for that peril, while ICL's
point estimate (+1.3%) does not carry a comparably large uncertainty band by
construction (ICL is a different, deterministic combination of paid+incurred
information, not perturbed by this Mack-paid-only simulation) - a caveat worth
stating plainly rather than implying the two are on equal footing.

`requirements-lock.txt` was also found to be a raw `pip freeze` of the host
machine's system Python (no virtual environment was active when it was first
generated), so alongside the real dependencies it listed ~75 entirely
unrelated Ubuntu OS/desktop packages (`aptdaemon`, `bcc`, `systemd-python`,
`ubuntu-pro-client`, `PyGObject`, and similar) - a real reproducibility-
hygiene defect, since the file is supposed to be a trustworthy record of what
this pipeline actually depends on. Regenerated by walking the real transitive
dependency closure of `requirements.txt`'s eight direct packages via
`importlib.metadata` (not another raw freeze, and not a manual guess at which
of the ~110 installed packages "look relevant"): 34 genuine packages remain.
`env_block.tex` was unaffected (the five version numbers it cites were always
correct) and regenerates identically from the cleaned file.

## Repository layout

```
├── .github/workflows/ci.yml  # runs tests + run_all.py on synthetic data on every push
├── config/
│   ├── default.yaml          # split cutoffs, hyperparameters - not derived from real data
│   └── seeds.yaml            # explicit 20-seed list for the multi-seed run
├── data/
│   ├── README.md              # expected schema; no real data
│   └── synthetic/             # generator + generated synthetic triangles
├── src/
│   ├── data_pipeline.py       # loading, scaling, tensorising, zero-leakage split
│   ├── seeding.py             # single source of truth for determinism
│   ├── train.py                # LSTM training loop, early stopping
│   ├── evaluate.py             # autoregressive rollout, metrics
│   ├── pipeline.py             # single-peril AND pooled multi-peril evaluation
│   ├── interpretability.py     # occlusion-based channel importance
│   ├── diagnostics.py          # pre/post-period link ratio comparison
│   ├── metrics.py              # per-unit Freq x Sev = BC identities
│   ├── classical/              # mack.py, icl.py, munich.py, bf.py
│   └── ml/                     # lstm.py, gbm.py
├── scripts/
│   ├── run_all.py              # single-seed full pipeline, both perils
│   ├── run_multiseed.py        # 20-seed single-peril LSTM variance experiment;
│   │                           # --rnn-type lstm|gru, --use-counts for the two
│   │                           # architecture/feature variants tested this round
│   ├── run_pooled_multiseed.py # 20-seed pooled multi-peril LSTM variance experiment
│   ├── run_pooled_ensemble.py  # averages the pooled sweep's 20 checkpoints' predictions
│   ├── make_figures.py         # figures from the same computed results
│   └── make_env_block.py       # env_block.tex from requirements-lock.txt
├── outputs/                    # gitignored; regenerate, don't commit
└── tests/
    ├── test_classical_models.py     # unit tests against a hand-worked example
    ├── test_metrics.py               # per-unit identity tests
    ├── test_interpretability.py      # occlusion harness sanity check
    ├── test_pooled_pipeline.py       # pooled architecture smoke + determinism
    ├── test_mack_windowed_bootstrap.py  # windowed Mack SE determinism + reconciliation
    ├── test_architecture_variants.py    # GRU cell and claim-count-feature smoke + determinism
    └── test_reconciliation.py        # integration tests against outputs/results.json
```

## Method summary

| Model                  | Information set                | Class                        |
| ---------------------- | ------------------------------ | ---------------------------- |
| Mack Paid Chain Ladder | Paid only                      | Classical, deterministic     |
| Bornhuetter-Ferguson   | Paid + exposure prior          | Classical, exposure-anchored |
| Incurred Chain Ladder  | Incurred (paid + case reserve) | Classical, deterministic     |
| Munich Chain Ladder    | Joint paid + case reserve      | Classical, joint             |
| Gradient Boosting      | Joint paid + case reserve      | Non-neural ML                |
| DeepTriangle LSTM      | Joint paid + case reserve      | Neural, autoregressive       |

Indexing convention throughout: `i` = accident quarter (0-indexed), `j` =
development quarter (0-indexed), `k = i + j` = calendar quarter. A cell is
observed only while `k` does not exceed the last available calendar quarter.

## Tests

```Shell
python scripts/run_all.py                 # writes outputs/results.json
python -m unittest discover -s tests -v
```

Only the standard library's `unittest` is required - no extra test-runner
dependency to install. Seven files (23 tests), two different kinds of check:

- `test_classical_models.py`, `test_metrics.py`, `test_interpretability.py`,
  `test_pooled_pipeline.py`, `test_mack_windowed_bootstrap.py`, and
  `test_architecture_variants.py` are unit tests against hand-computable
  examples or internal-consistency checks (including the dissertation's own
  worked Table 1), independent of any real claims data file - they run against
  the bundled synthetic data instead.
- `test_reconciliation.py` is an integration suite: it checks internal
  identities against whatever is currently in `outputs/results.json`. It does
  not assert against a hardcoded target answer, so it fails honestly if a
  future code change breaks a number's derivation, and it is what caught two
  real bugs (an LSTM parameter-count formula and a reserve-decomposition
  scoping error) during this pipeline's own development.

## Continuous integration

`.github/workflows/ci.yml` runs the full test suite plus `run_all.py` and
`make_figures.py` on every push and pull request, entirely against the
bundled synthetic data - it never receives a `--data-dir` argument, so it has
no access to and makes no use of anyone's real data, on this repository or a
fork of it. Its only purpose is to prove, automatically and on a clean
checkout, that the pipeline installs and runs end to end exactly as this
README describes. If you fork this repo and want to point the pipeline at
your own real data, that is a separate, manual, local step exactly as
described under "Reproducing the dissertation results" above - CI does not
do this for you and is not designed to, since a workflow that could pull in
arbitrary external data on every push is not something this repository wants
to enable by default.

## Licence

No `LICENSE` file is included yet. One (MIT) will be added once finalised.
Until then, all rights are reserved by default.

This repository contains no real claims data of any kind.
