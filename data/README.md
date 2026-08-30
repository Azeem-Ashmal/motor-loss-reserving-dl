# Data

This repository contains no proprietary claims data. Results in the dissertation
were produced on a proprietary industry motor-claims database that cannot be
redistributed. See `data/synthetic/` for a runnable substitute.

## Expected input format

Every model in `src/` consumes two CSV files per peril:

- `paid_qtr_triangle.csv` - cumulative paid losses, `C_{i,j}`
- `outstanding_qtr_triangle.csv` - case reserves / BALOS, `B_{i,j}`

Both files share the same shape and column layout:

| Column | Type | Meaning |
|---|---|---|
| `Cohort` | string | Accident quarter label, e.g. `"2010 Q1"` |
| `Dev_0` ... `Dev_{N-1}` | float | Cumulative paid (or outstanding) value at that development quarter |

- One row per accident quarter, in chronological order (row 0 = the earliest cohort).
- `Dev_j` for accident quarter `i` is unobserved (and must be `NaN`, i.e. blank in
  the CSV) whenever `i + j` exceeds the last available calendar quarter - the
  upper-right triangle. This is what makes it a triangle rather than a square.
- Values are in the same monetary unit throughout a file (the dissertation uses RM).
- Indexing is 0-indexed throughout the pipeline: `i` = accident quarter,
  `j` = development quarter, `k = i + j` = calendar quarter.

Directory layout expected by `scripts/run_all.py --data-dir <dir>`:

```
<dir>/
├── theft/
│   ├── paid_qtr_triangle.csv
│   ├── outstanding_qtr_triangle.csv
│   ├── exposure_qtr.csv              (optional - enables Bornhuetter-Ferguson)
│   └── claims_count_qtr_triangle.csv (optional - enables --use-counts)
└── windscreen/
    ├── paid_qtr_triangle.csv
    ├── outstanding_qtr_triangle.csv
    ├── exposure_qtr.csv              (optional)
    └── claims_count_qtr_triangle.csv (optional)
```

`exposure_qtr.csv` is optional and only needed for the Bornhuetter-Ferguson
baseline. Two columns, one row per accident quarter, same row count and order
as the triangle files:

| Column | Type | Meaning |
|---|---|---|
| `Cohort` | string | Accident quarter label, matching the triangle files |
| `Exposure` | float | Earned exposure for that accident quarter (e.g. vehicle-years) |

If this file is missing, `run_all.py` skips Bornhuetter-Ferguson and runs
every other model normally.

`claims_count_qtr_triangle.csv` is optional and only used by
`run_multiseed.py --use-counts` (see the main README's "Verified findings" -
this was tested as a candidate fix for the LSTM's underperformance and made it
significantly worse, not better, but the code path is real and tested).
Same shape and cumulative convention as `paid_qtr_triangle.csv`: one row per
accident quarter, `Dev_j` columns holding cumulative reported claim counts,
NaN in the unobserved upper-right triangle.

## Synthetic substitute

`data/synthetic/generate.py` writes exactly this layout under `data/synthetic/`,
with round, seeded, obviously-fake numbers shaped like a low-frequency/
high-severity peril (theft) and a high-frequency/low-severity peril
(windscreen). Running `scripts/run_all.py` with no `--data-dir` argument uses
this data automatically, so the pipeline is fully runnable by anyone without
access to real claims data - it exercises every model end to end, but the
numbers it produces are not the dissertation's results.

## Reproducing the dissertation results

If you have access to an equivalent triangle extract in the same shape, lay it out
as above and run:

```
python scripts/run_all.py --data-dir /path/to/your/data --seed 42
```

`outputs/results.json` will contain the same fields, computed the same way,
regardless of which data directory you point at.
