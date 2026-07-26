# Example 3: Optimizer Comparison (varpro vs basin_hopping)

This optional example compares the default local optimizer (`varpro`) against
the global optimizer (`basin_hopping`) on the same pseudo-3D dataset.

## Why This Example

Use this only when some clusters converge poorly with the default optimizer and
you want to test whether global search improves fit quality. For routine fits,
use Example 2.

## Quick Start

Run both methods and print a chi-squared comparison:

```bash
bash run.sh both
```

Other modes:

```bash
bash run.sh local
bash run.sh basin
```

## Manual Commands

Local baseline:

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --optimizer varpro \
  --output Fits-local
```

Global optimization:

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --optimizer basin_hopping \
  --output Fits-basin
```

Each command writes to a timestamped run under the selected output root.

## Compare Results

Inspect:

- `summary/fit.json`
- `tables/parameters.csv`

The script prints a simple comparison using values from `summary/fit.json`.
