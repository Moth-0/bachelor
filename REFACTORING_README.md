# Deuteron SVM Refactoring: Automated Parameter Sweeps & Plotting

## Overview

This refactored system enables **fully automated parameter sweeps and plotting** for the quantum mechanics deuteron solver. Instead of manually running simulations and extracting data, you can now:

1. **Sweep parameters** (b_form, S, basis_size) with a single Make command
2. **Automatically aggregate** results into CSV files
3. **Generate plots** directly from aggregated data
4. **Track reproducibility** with full metadata (timestamps, parameters, etc.)

---

## Quick Start

### Single Run with CSV Output
```bash
./deu -b_range 2.5 -b_form 1.5 -S 38.4 --output-csv results/test/run_001.csv
```

Produces a CSV with metadata and convergence history.

### Run a Parameter Sweep
```bash
make sweep-b_form          # Sweep b_form parameter (1.0-2.5, 10 steps)
make sweep-S               # Sweep S coupling (20-100 MeV, 15 steps)
make sweep-basis-size      # Sweep basis size limit (10-60, 10 steps)
```

### Generate Plots from Sweep Results
```bash
make plot-energy-vs-b_form   # Plot Energy vs b_form
make plot-energy-vs-S        # Plot Energy vs S coupling
```

### Run Full Experiment (All Sweeps + Plots)
```bash
make full-experiment
```

This runs all parameter sweeps and generates all plots in one command.

---

## Implementation Details

### Phase 1: CSV Output Enhancement

**File**: `qm/csv_writer.h`, modified `deu.cc`

The core solver (`deu`) now supports CSV output with full metadata:

```bash
./deu \
  -b_range 2.5 -b_form 1.5 -S 38.4 \
  --output-csv results/sweep_test/run_1.csv \
  --max-basis-size 50
```

**CSV Output Format**:
```
# METADATA: b_range=2.500000
# METADATA: b_form=1.500000
# METADATA: S=38.400000
# METADATA: max_basis_size=50
# METADATA: timestamp=2026-05-07T18:15:22
iteration,energy_mev,kinetic_mev,radius_fm,basis_size
0,-8.10952,29.77616,1.50677,12
1,-8.15289,30.10423,1.50107,12
...
-1,-8.00395,31.66627,1.45136,39
```

Last row has `iteration=-1` as FINAL marker.

### Phase 2: Parameter Sweep Orchestration

**File**: `scripts/sweep.py`

Automatically runs multiple simulations with different parameters:

```bash
python3 scripts/sweep.py --scan b_form \
  --b_range 2.5 --S 38.4 \
  --b_form_min 1.0 --b_form_max 2.5 --b_form_steps 10 \
  --jobs 4
```

**Features**:
- Parallel job execution (configurable via `--jobs`)
- Automatic parameter combination generation (numpy linspace)
- Result aggregation from all runs
- Metadata tracking (timestamps, parameters, success/failure)
- Organized output: `results/energy_sweep_b_form/`

**Output**:
```
results/energy_sweep_b_form/
├── run_abc123.csv          # Individual run
├── run_def456.csv          # Individual run
├── aggregated.csv          # Merged results (sorted by parameter)
├── metadata.json           # Sweep metadata
└── sweep.log               # Execution log
```

### Phase 3: Makefile Orchestration

**File**: `Makefile` (new targets added)

Simple Make targets for common workflows:

```makefile
make sweep-b_form          # Predefined b_form sweep
make sweep-S               # Predefined S sweep
make sweep-basis-size      # Predefined basis size sweep
make plot-energy-vs-b_form # Generate plots
make plot-energy-vs-S      # Generate plots
make all-sweeps            # Run all three sweeps
make all-plots             # Generate all plots
make full-experiment       # Everything
```

### Phase 6: Plotting Scripts

**File**: `scripts/plot_results.py`

Generate publication-quality plots from sweep results:

```bash
python3 scripts/plot_results.py energy_sweep_b_form
python3 scripts/plot_results.py energy_sweep_S
python3 scripts/plot_results.py basis_size_convergence
```

**Features**:
- Matplotlib-based plotting
- Dual-axis plots (energy + radius)
- Reference lines (target values)
- Automatic data sorting by parameter
- High-DPI output (150 DPI PNG)

**Plots Generated**:
- `results/energy_sweep_b_form/plot.png` — Energy & radius vs b_form
- `results/energy_sweep_S/plot.png` — Energy & radius vs S
- `results/basis_size_convergence/convergence.png` — Convergence curves

---

## Data Organization

Results are organized by scan type:

```
results/
├── energy_sweep_b_form/
│   ├── run_*.csv           # Individual runs
│   ├── aggregated.csv      # All runs merged
│   ├── metadata.json       # Sweep metadata
│   ├── plot.png            # Generated plot
│   └── sweep.log           # Execution log
├── energy_sweep_S/
│   └── [same structure]
├── basis_size_convergence/
│   └── [same structure]
└── raw/
    └── [one-off runs if --output-csv used directly]
```

---

## Reproducibility Features

Every sweep captures **full metadata** for reproducibility:

### Metadata Rows in CSV
```
# METADATA: b_range=2.5
# METADATA: b_form=1.5
# METADATA: S=38.4
# METADATA: max_basis_size=50
# METADATA: timestamp=2026-05-07T18:15:22
```

### Metadata JSON File
```json
{
  "scan_type": "b_form",
  "timestamp": "2026-05-07T18:15:22",
  "fixed_params": {
    "b_range": 2.5,
    "S": 38.4
  },
  "sweep_params": {
    "b_form_min": 1.0,
    "b_form_max": 2.5,
    "b_form_steps": 10
  },
  "total_runs": 10,
  "successful_runs": 10
}
```

---

## Advanced Usage

### Custom Parameter Ranges

Override sweep parameters directly:

```bash
# Sweep b_form from 1.0 to 3.0 with 20 steps
make sweep-b_form B_FORM_MIN=1.0 B_FORM_MAX=3.0 B_FORM_STEPS=20

# Use more parallel jobs
make sweep-S JOBS=8

# Single custom run
./deu -b_range 2.5 -b_form 1.2 -S 50.0 --output-csv results/custom/test.csv
```

### Parallel Execution

Sweeps run in parallel by default (4 jobs). Control with:

```bash
# Sequential execution
python3 scripts/sweep.py ... --jobs 1

# Use all CPU cores (12 on this machine)
python3 scripts/sweep.py ... --jobs 12
```

### Manual Aggregation

If you run individual simulations separately, you can aggregate manually:

```bash
# After running multiple sweeps
ls results/energy_sweep_b_form/*.csv
# → run_1.csv, run_2.csv, ...
# → aggregated.csv is created automatically by sweep.py
```

---

## Troubleshooting

### CSV File Not Created
```bash
# Ensure results directory exists
mkdir -p results/test

# Test deu with explicit path
./deu -b_range 2.5 -b_form 1.5 -S 38.4 --output-csv results/test/run_001.csv

# Check if file was created
ls -l results/test/run_001.csv
head results/test/run_001.csv
```

### Sweep Failing with Permission Error
```bash
# Make scripts executable
chmod +x scripts/*.py

# Ensure deu is built
make clean && make deu
```

### Plots Not Generating
```bash
# Install matplotlib if needed
pip install matplotlib numpy

# Test plot script
python3 scripts/plot_results.py energy_sweep_b_form

# If CSV doesn't exist, run sweep first
make sweep-b_form
make plot-energy-vs-b_form
```

---

## Integration with Existing Code

**No changes to core physics:**
- All SVM optimization (SVM.h) unchanged
- Operators unchanged (qm/operators.h)
- Gaussian basis unchanged (qm/gaussian.h)
- Only I/O and orchestration layer modified

**Backward compatibility:**
- Old `convergence.data` format still generated
- Old `basis_final.txt` still saved
- All existing tests in `test.cc` still work

---

## Future Enhancements

Potential additions (not yet implemented):

1. **Basis Size Sweeps**: Add `--max-basis-size` flag to control basis optimization limits
2. **Error Analysis**: Track uncertainty bands in plots
3. **SQLite Results Database**: Searchable query interface for all results
4. **Configuration Files**: YAML/JSON for common experiment templates
5. **Automated Optimization**: Binary search for optimal S parameter

---

## Files Modified/Created

**Created**:
- `qm/csv_writer.h` — CSV output class
- `scripts/sweep.py` — Parameter sweep orchestrator
- `scripts/plot_results.py` — Plotting utilities
- `results/` directory — Results organization

**Modified**:
- `deu.cc` — Added CSV output, command-line args
- `Makefile` — Added sweep and plot targets

**Unchanged**:
- All `qm/` headers (physics engine)
- `SVM.h` (optimization algorithm)
- `test.cc` (test suite)
- All parameter files

---

## Example Workflow

```bash
# 1. Build the code
make clean && make

# 2. Run a quick sweep (2-3 minutes)
python3 scripts/sweep.py --scan b_form \
  --b_range 2.5 --S 38.4 \
  --b_form_min 1.4 --b_form_max 1.6 --b_form_steps 2 \
  --jobs 1

# 3. View results
cat results/energy_sweep_b_form/aggregated.csv

# 4. Generate plot
python3 scripts/plot_results.py energy_sweep_b_form

# 5. View plot
open results/energy_sweep_b_form/plot.png  # On macOS
# or
xdg-open results/energy_sweep_b_form/plot.png  # On Linux
```

---

## Support

For issues or questions:
1. Check CSV file format: `head -10 results/*/run_*.csv`
2. Check sweep metadata: `cat results/*/metadata.json`
3. Verify plots with: `python3 scripts/plot_results.py <scan_type>`

