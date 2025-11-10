#!/usr/bin/env python3
"""
Create plots for Closest Pair analysis from timing_results.xlsx
Run this in the same folder as your Excel file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Read the data
bf_data = pd.read_excel('timing_results.xlsx', sheet_name='Brute Force')
dc_data = pd.read_excel('timing_results.xlsx', sheet_name='Divide and Conquer')

# Extract input sizes and calculate means/stds from trials
n_values = bf_data['Input Size (n)'].values
bf_means = bf_data.iloc[:, 1:11].mean(axis=1).values * 1000  # Convert to milliseconds
bf_stds = bf_data.iloc[:, 1:11].std(axis=1).values * 1000    # Convert to milliseconds
dc_means = dc_data.iloc[:, 1:11].mean(axis=1).values * 1000  # Convert to milliseconds
dc_stds = dc_data.iloc[:, 1:11].std(axis=1).values * 1000    # Convert to milliseconds

print("Creating plots...")
print(f"Input sizes: {n_values}")
print(f"BF means (ms): {bf_means}")
print(f"DC means (ms): {dc_means}")

# ============================================================================
# PLOT 1: Brute Force Algorithm (similar to your "Foo Algorithm" example)
# ============================================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Scatter individual trials (blue dots)
for i in range(1, 11):
    trial_data = bf_data[f'Trial {i}'].values * 1000  # Convert to milliseconds
    ax1.scatter(n_values, trial_data, alpha=0.4, s=40, color='steelblue', label='Instance Time' if i == 1 else '')

# Plot average line (orange)
ax1.plot(n_values, bf_means, 'o-', color='darkorange', linewidth=3, 
         markersize=10, label='Average Time', zorder=5)

# Fit quadratic trendline
coeffs = np.polyfit(n_values, bf_means, 2)
n_fit = np.linspace(n_values[0], n_values[-1], 100)
quadratic_fit = coeffs[0] * n_fit**2 + coeffs[1] * n_fit + coeffs[2]
ax1.plot(n_fit, quadratic_fit, '--', color='cornflowerblue', linewidth=2.5, 
         label='Quadratic trendline', zorder=4)

ax1.set_xlabel('Number of Points (n)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Runtime (milliseconds)', fontsize=13, fontweight='bold')
ax1.set_title('Brute Force Algorithm Running Time', fontsize=15, fontweight='bold')
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_xlim(0, max(n_values) + 50)
ax1.set_ylim(0, max(bf_means) * 1.1)

plt.tight_layout()
plt.savefig('brute_force_plot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: brute_force_plot.png")

# ============================================================================
# PLOT 2: Divide and Conquer Algorithm
# ============================================================================
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Scatter individual trials (green dots)
for i in range(1, 11):
    trial_data = dc_data[f'Trial {i}'].values * 1000  # Convert to milliseconds
    ax2.scatter(n_values, trial_data, alpha=0.4, s=40, color='mediumseagreen', label='Instance Time' if i == 1 else '')

# Plot average line (orange)
ax2.plot(n_values, dc_means, 'o-', color='darkorange', linewidth=3, 
         markersize=10, label='Average Time', zorder=5)

# Fit n log n trendline
def nlogn(n, a, b):
    return a * n * np.log(n) + b

popt, _ = curve_fit(nlogn, n_values[1:], dc_means[1:])  # Skip n=1
nlogn_fit = nlogn(n_fit, *popt)
ax2.plot(n_fit, nlogn_fit, '--', color='cornflowerblue', linewidth=2.5, 
         label='n log n trendline', zorder=4)

ax2.set_xlabel('Number of Points (n)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Runtime (milliseconds)', fontsize=13, fontweight='bold')
ax2.set_title('Divide-and-Conquer Algorithm Running Time', fontsize=15, fontweight='bold')
ax2.legend(fontsize=11, loc='upper left')
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax2.set_xlim(0, max(n_values) + 50)
ax2.set_ylim(0, max(dc_means) * 1.1)

plt.tight_layout()
plt.savefig('divide_conquer_plot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: divide_conquer_plot.png")

# ============================================================================
# PLOT 3: Direct Comparison (Log Scale)
# ============================================================================
fig3, ax3 = plt.subplots(figsize=(10, 6))

ax3.plot(n_values, bf_means, 'o-', color='steelblue', linewidth=2.5, 
         markersize=10, label='Brute Force', marker='s')
ax3.plot(n_values, dc_means, 'o-', color='mediumseagreen', linewidth=2.5, 
         markersize=10, label='Divide-and-Conquer', marker='^')

# Add error bars
ax3.errorbar(n_values, bf_means, yerr=bf_stds, fmt='none', 
             ecolor='steelblue', alpha=0.3, capsize=5)
ax3.errorbar(n_values, dc_means, yerr=dc_stds, fmt='none', 
             ecolor='mediumseagreen', alpha=0.3, capsize=5)

ax3.set_xlabel('Number of Points (n)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Runtime (milliseconds)', fontsize=13, fontweight='bold')
ax3.set_title('Algorithm Comparison', fontsize=15, fontweight='bold')
ax3.legend(fontsize=12)
ax3.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
ax3.set_yscale('log')  # Log scale to show both clearly

plt.tight_layout()
plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comparison_plot.png")

# ============================================================================
# PLOT 4: Speedup Factor
# ============================================================================
fig4, ax4 = plt.subplots(figsize=(10, 6))

speedup = bf_means / dc_means
ax4.plot(n_values, speedup, 'o-', color='purple', linewidth=3, markersize=10)

ax4.set_xlabel('Number of Points (n)', fontsize=13, fontweight='bold')
ax4.set_ylabel('Speedup Factor (Brute Force / D&C)', fontsize=13, fontweight='bold')
ax4.set_title('Performance Speedup of Divide-and-Conquer', fontsize=15, fontweight='bold')
ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Add value labels
for n, speed in zip(n_values[1:], speedup[1:]):  # Skip n=1
    ax4.annotate(f'{speed:.1f}x', 
                xy=(n, speed), 
                xytext=(0, 10), 
                textcoords='offset points',
                fontsize=10,
                ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('speedup_plot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: speedup_plot.png")

print("\n" + "="*60)
print("SUCCESS! All plots created:")
print("  1. brute_force_plot.png")
print("  2. divide_conquer_plot.png")
print("  3. comparison_plot.png")
print("  4. speedup_plot.png")
print("="*60)

# Print statistics
print("\nKEY STATISTICS:")
print(f"  Speedup at n=500: {speedup[-1]:.1f}x faster")
print(f"  BF runtime at n=500: {bf_means[-1]:.2f} milliseconds")
print(f"  D&C runtime at n=500: {dc_means[-1]:.4f} milliseconds")