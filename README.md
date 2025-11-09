# Closest Pair of Points

**Authors:** Alex Bennett, Ella Kocher, Yina Tang  
**Course:** COMP 422, Fall 2025

---

## Installation

Install required packages:
```bash
pip install numpy openpyxl
```

If `pip` is not recognized:
```bash
python -m pip install numpy openpyxl
```

---

## Commands

### Basic Usage

**Run with default settings (5000 points):**
```bash
python closest_pair.py
```

**Run with custom number of points:**
```bash
python closest_pair.py --num-points 1000
python closest_pair.py -np 1000
```

### Data Collection

**Collect comprehensive timing data:**
```bash
python closest_pair.py --collect-data
python closest_pair.py -cd
```

This tests both algorithms with n = 1, 200, 400, 600, 800, 1000 points (10 trials each) and saves results to `timing_results.xlsx`.

**Specify custom output filename:**
```bash
python closest_pair.py --collect-data --output-file results.xlsx
python closest_pair.py -cd -o results.xlsx
```

### Help

**View all options:**
```bash
python closest_pair.py --help
python closest_pair.py -h
```

---

## Command Reference

| Command | Short | Description |
|---------|-------|-------------|
| `--num-points N` | `-np N` | Number of points to generate (default: 5000) |
| `--collect-data` | `-cd` | Run timing analysis and save to Excel |
| `--output-file FILE` | `-o FILE` | Excel output filename (default: timing_results.xlsx) |
| `--help` | `-h` | Show help message |

---

## Output

**Single test mode** displays:
- Closest pair found by each algorithm
- Minimum distance between the pair
- Runtime for each algorithm

**Data collection mode** creates:
- `timing_results.xlsx` with three sheets:
  - Brute Force: Trial times and statistics
  - Divide and Conquer: Trial times and statistics
  - Summary: Comparative analysis with speedup factors

---

## Optional: Generate Plots

To create performance visualization plots:

1. Install additional packages:
```bash
pip install matplotlib scipy pandas
```

2. Run the plot generator:
```bash
python make_plots.py
```

This creates 4 PNG files showing algorithm performance comparisons.