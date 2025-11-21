# Example 5: Batch Processing Multiple Datasets

## Overview

This example demonstrates efficient workflows for processing multiple NMR spectra in batch mode. Batch processing is essential for:
- Analyzing replicate measurements
- Processing time-series experiments
- Comparing multiple samples or conditions
- High-throughput screening

**Note:** This is a template demonstrating workflow patterns. Adapt the scripts for your specific dataset structure.

## When You Need Batch Processing

Use batch processing when you have:

✅ **Multiple related datasets:**
- Time series (e.g., kinetics experiments)
- Replicate measurements
- Different conditions (temperature, pH, etc.)
- Different samples with same peak assignments

✅ **Consistent peak lists:**
- Same or similar peak assignments across datasets
- Can use shared peak list or sample-specific lists

✅ **Need for automation:**
- Manual processing is too time-consuming
- Want consistent analysis across all data
- Need to reprocess with different parameters

## Batch Processing Strategies

### Strategy 1: Shared Peak List

**When to use:**
- Same sample under different conditions
- Replicate measurements
- Time series of same sample

**Example:**
```bash
for spectrum in data/experiment*.ft2; do
    output=$(basename "$spectrum" .ft2)
    peakfit fit "$spectrum" data/shared_peaks.list \
        --output "Results-$output/"
done
```

### Strategy 2: Individual Peak Lists

**When to use:**
- Different samples
- Peak positions vary between datasets
- Sample-specific assignments

**Example:**
```bash
for spectrum in data/*.ft2; do
    base=$(basename "$spectrum" .ft2)
    peakfit fit "$spectrum" "data/${base}.list" \
        --output "Results-$base/"
done
```

### Strategy 3: Configuration File Template

**When to use:**
- Complex parameter sets
- Need reproducibility
- Different parameters per dataset

**Example:**
```bash
for config in configs/*.toml; do
    peakfit fit data/spectrum.ft2 data/peaks.list \
        --config "$config" \
        --output "Results-$(basename $config .toml)/"
done
```

## Example Workflow: Time Series

### Scenario

You have CEST experiments at different time points:
- `timepoint_0h.ft2`
- `timepoint_1h.ft2`
- `timepoint_2h.ft2`
- ... up to `timepoint_24h.ft2`

All use the same peak list: `peaks.list`

### Batch Script

```bash
#!/bin/bash
# Process time series

# Parameters
PEAKLIST="data/peaks.list"
ZVALUES="data/b1_offsets.txt"
OUTPUT_BASE="Results"

# Process each timepoint
for timepoint in data/timepoint_*.ft2; do
    # Extract timepoint name
    name=$(basename "$timepoint" .ft2)

    echo "Processing $name..."

    # Run PeakFit
    peakfit fit "$timepoint" "$PEAKLIST" \
        --z-values "$ZVALUES" \
        --output "${OUTPUT_BASE}-${name}/"

    # Check success
    if [ $? -eq 0 ]; then
        echo "  ✓ $name complete"
    else
        echo "  ✗ $name FAILED"
    fi
done

echo
echo "Batch processing complete!"
echo "Results in ${OUTPUT_BASE}-timepoint_*/"
```

### Running the Workflow

```bash
bash batch_process.sh
```

## Organizing Results

### Directory Structure

```
project/
├── data/
│   ├── timepoint_0h.ft2
│   ├── timepoint_1h.ft2
│   ├── ...
│   ├── peaks.list
│   └── b1_offsets.txt
├── Results-timepoint_0h/
│   ├── shifts.list
│   ├── peakfit.log
│   └── *.out
├── Results-timepoint_1h/
│   ├── shifts.list
│   ├── peakfit.log
│   └── *.out
├── ...
└── batch_process.sh
```

## Collecting Results

### Extract Chemical Shifts

Collect fitted shifts from all timepoints:

```bash
#!/bin/bash
# collect_shifts.sh

echo "Timepoint,Peak,F1,F2" > all_shifts.csv

for dir in Results-timepoint_*/; do
    timepoint=$(basename "$dir" | sed 's/Results-timepoint_//' | sed 's/h//')

    # Extract shifts
    awk -v tp="$timepoint" 'NR>1 {print tp","$1","$2","$3}' \
        "${dir}/shifts.list" >> all_shifts.csv
done

echo "Collected shifts saved to: all_shifts.csv"
```

### Extract Intensities

Collect CEST intensities for specific peak:

```bash
#!/bin/bash
# collect_cest.sh

PEAK="10N-HN"

# Header
echo -n "Z-value" > "cest_${PEAK}.csv"
for dir in Results-timepoint_*/; do
    tp=$(basename "$dir" | sed 's/Results-timepoint_//')
    echo -n ",$tp" >> "cest_${PEAK}.csv"
done
echo "" >> "cest_${PEAK}.csv"

# Data rows
# Get Z-values from first file
awk 'NR>1 && !/^#/ {print $1}' "Results-timepoint_0h/${PEAK}.out" | \
while read z; do
    echo -n "$z" >> "cest_${PEAK}.csv"

    for dir in Results-timepoint_*/; do
        intensity=$(awk -v z="$z" '$1==z {print $2}' "${dir}/${PEAK}.out")
        echo -n ",$intensity" >> "cest_${PEAK}.csv"
    done
    echo "" >> "cest_${PEAK}.csv"
done

echo "CEST profiles saved to: cest_${PEAK}.csv"
```

## Parallel Processing

For faster batch processing, run multiple fits in parallel:

```bash
#!/bin/bash
# parallel_process.sh

# Number of parallel jobs
NJOBS=4

export PEAKLIST="data/peaks.list"
export ZVALUES="data/b1_offsets.txt"

# Function to process one file
process_file() {
    spectrum=$1
    name=$(basename "$spectrum" .ft2)

    peakfit fit "$spectrum" "$PEAKLIST" \
        --z-values "$ZVALUES" \
        --output "Results-${name}/" \
        &> "logs/${name}.log"
}

export -f process_file

# Create log directory
mkdir -p logs

# Process files in parallel using GNU parallel
find data/ -name "*.ft2" | \
    parallel -j $NJOBS process_file {}

# Or use xargs if GNU parallel isn't available:
# find data/ -name "*.ft2" | xargs -P $NJOBS -I {} bash -c 'process_file "$@"' _ {}
```

## Quality Control

### Check Success Rate

```bash
#!/bin/bash
# check_results.sh

total=0
success=0
failed=0

for dir in Results-*/; do
    total=$((total + 1))

    if grep -q "Fitting complete" "${dir}/peakfit.log"; then
        success=$((success + 1))
    else
        failed=$((failed + 1))
        echo "FAILED: $dir"
    fi
done

echo
echo "Summary:"
echo "  Total:   $total"
echo "  Success: $success"
echo "  Failed:  $failed"
echo "  Rate:    $((success * 100 / total))%"
```

### Compare Results

Check consistency across datasets:

```bash
#!/bin/bash
# compare_shifts.sh

echo "Checking chemical shift consistency..."

# Extract first dataset as reference
ref="Results-timepoint_0h/shifts.list"

for dir in Results-timepoint_*/; do
    shifts="${dir}/shifts.list"

    if [ "$shifts" != "$ref" ]; then
        # Compare shifts (allowing small differences)
        # This is a simplified check - adapt for your needs
        diff <(awk '{print $1,$2,$3}' "$ref" | sort) \
             <(awk '{print $1,$2,$3}' "$shifts" | sort) | \
        grep "^>" | wc -l
    fi
done
```

## Error Handling

### Robust Batch Script

```bash
#!/bin/bash
# robust_batch.sh

set -e  # Exit on error in main script
trap 'echo "Error on line $LINENO"' ERR

# Log file
LOGFILE="batch_$(date +%Y%m%d_%H%M%S).log"

{
    echo "Batch processing started: $(date)"

    for spectrum in data/*.ft2; do
        name=$(basename "$spectrum" .ft2)

        echo "Processing $name..."

        # Try fitting with error handling
        if peakfit fit "$spectrum" data/peaks.list \
            --output "Results-${name}/" 2>&1; then
            echo "  ✓ $name succeeded"
        else
            echo "  ✗ $name failed (exit code $?)"
            # Continue with other files
            continue
        fi
    done

    echo "Batch processing completed: $(date)"

} | tee "$LOGFILE"
```

## Advanced Patterns

### Conditional Processing

Skip files that have already been processed:

```bash
for spectrum in data/*.ft2; do
    name=$(basename "$spectrum" .ft2)
    output_dir="Results-${name}"

    # Skip if already processed
    if [ -f "${output_dir}/shifts.list" ]; then
        echo "Skipping $name (already processed)"
        continue
    fi

    echo "Processing $name..."
    peakfit fit "$spectrum" data/peaks.list --output "$output_dir/"
done
```

### Parameter Sweeps

Test different contour factors:

```bash
for contour in 3.0 5.0 7.0 10.0; do
    echo "Testing contour factor: $contour"

    peakfit fit data/spectrum.ft2 data/peaks.list \
        --contour-factor $contour \
        --output "Results-contour${contour}/"
done

# Compare results
for dir in Results-contour*/; do
    echo "$dir:"
    grep "Successful" "${dir}/peakfit.log"
done
```

## Visualization

### Plot Time Series

```python
import matplotlib.pyplot as plt
import pandas as pd

# Load collected data
shifts = pd.read_csv('all_shifts.csv')

# Plot one peak over time
peak = '10N-HN'
peak_data = shifts[shifts['Peak'] == peak]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(peak_data['Timepoint'], peak_data['F1'], 'o-')
plt.xlabel('Time (h)')
plt.ylabel('F1 (ppm)')
plt.title(f'{peak} - F1 shift')

plt.subplot(1, 2, 2)
plt.plot(peak_data['Timepoint'], peak_data['F2'], 'o-')
plt.xlabel('Time (h)')
plt.ylabel('F2 (ppm)')
plt.title(f'{peak} - F2 shift')

plt.tight_layout()
plt.savefig(f'{peak}_timecourse.pdf')
```

## Troubleshooting

### "Some files failed to process"

Check log files:
```bash
for log in logs/*.log; do
    if grep -q "Error" "$log"; then
        echo "Errors in: $log"
        grep "Error" "$log"
    fi
done
```

### "Results are inconsistent"

1. **Check data quality:** Are all spectra comparable?
2. **Verify peak lists:** Are assignments correct for each spectrum?
3. **Check processing:** Same parameters for all?

### "Processing is too slow"

1. **Use parallel processing:** See parallel_process.sh example
2. **Optimize parameters:** Reduce refinement iterations for speed
3. **Use configuration files:** Avoid redundant validation

## Next Steps

1. **Adapt scripts for your data structure**
2. **Test on a few files first**
3. **Add dataset-specific quality checks**
4. **Create visualization scripts**
5. **Document your workflow**

## Reference

### Quick Script Templates

**Simple batch:**
```bash
for f in data/*.ft2; do
    peakfit fit "$f" peaks.list --output "$(basename $f .ft2)/"
done
```

**With logging:**
```bash
for f in data/*.ft2; do
    peakfit fit "$f" peaks.list --output "$(basename $f .ft2)/" \
        2>&1 | tee "$(basename $f .ft2).log"
done
```

**Parallel:**
```bash
ls data/*.ft2 | parallel -j4 \
    'peakfit fit {} peaks.list --output {/.}/'
```

## Additional Resources

- **[Main Examples README](../README.md)** - Overview of all examples
- **[Example 2: Advanced Fitting](../02-advanced-fitting/)** - Single dataset fitting
- **[Optimization Guide](../../docs/optimization_guide.md)** - Performance tuning
- **[GitHub Issues](https://github.com/gbouvignies/PeakFit/issues)** - Get help

---

**Questions about batch processing?** Open an issue at https://github.com/gbouvignies/PeakFit/issues
