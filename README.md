# Closest Pair of Points

**Authors:** Alex Bennett, Ella Kocher, Yina Tang
**Course:** COMP 422, Fall 2025

---

## Quick Start (Using algorithm.exe)

### Installation

Install required packages:
```bash
pip install numpy openpyxl
```

If `pip` is not recognized:
```bash
python -m pip install numpy openpyxl
```

### Running algorithm.exe

**Run with default settings:**
```bash
algorithm.exe
```

This tests both brute force and divide-and-conquer algorithms with input sizes:
- n = 1, 50, 100, 200, 400, 800, 1500, 2000, 3000 points
- 10 trials per input size
- Results saved to `timing_results.xlsx`

**Specify custom output filename:**
```bash
algorithm.exe --output-file my_results.xlsx
algorithm.exe -o my_results.xlsx
```

### Output

The executable creates `timing_results.xlsx` (or your specified filename) with three sheets:
- **Brute Force**: Individual trial times and statistics for each input size
- **Divide and Conquer**: Individual trial times and statistics for each input size
- **Summary**: Comparative analysis with mean times, standard deviations, and speedup factors

### Command Reference

| Command | Short | Description |
|---------|-------|-------------|
| `--output-file FILE` | `-o FILE` | Excel output filename (default: timing_results.xlsx) |

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

---

## Deprecated: Using algorithm.py Directly

> **Note:** The following commands are deprecated. Use `algorithm.exe` instead (see Quick Start section above).

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

### Command Reference (Deprecated)

| Command | Short | Description |
|---------|-------|-------------|
| `--num-points N` | `-np N` | Number of points to generate (default: 5000) |
| `--collect-data` | `-cd` | Run timing analysis and save to Excel |
| `--output-file FILE` | `-o FILE` | Excel output filename (default: timing_results.xlsx) |
| `--help` | `-h` | Show help message |

### Output (Deprecated)

**Single test mode** displays:
- Closest pair found by each algorithm
- Minimum distance between the pair
- Runtime for each algorithm

**Data collection mode** creates:
- `timing_results.xlsx` with three sheets:
  - Brute Force: Trial times and statistics
  - Divide and Conquer: Trial times and statistics
  - Summary: Comparative analysis with speedup factors