__author__ = "Alex Bennet, Ella Kocher, Yina Tang"
__description__ = "Implementation of closest pair algorithms. (Project 1 for COMP 422, Fall 2025)"

import argparse
from time import time
import numpy as np  # run `pip install numpy` in this directory in your console/terminal if you don't have it already
from numpy.typing import NDArray


def brute_force(points: NDArray) -> tuple[tuple[tuple[float, float]], float]:
    """Compute the minimum distance between points using brute-force approach.

    Arguments
    ---------
        points: NDArray
            An array of shape (n, d) where n is the number of points and d is the dimension.

    Returns
    -------
        closest_points: tuple[tuple[float, float]])
            An array of shape (2, d), i.e., the pair of points with the minimum distance.
        min_dist: float
            The minimum distance between the closest pair of points.
    """
    min_dist = float('inf')
    closest_points = None
    n = points.shape[0]
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points[i] - points[j])
            if dist < min_dist:
                min_dist = dist
                closest_points = (points[i].tolist(), points[j].tolist())

    return closest_points, min_dist


def divide_and_conquer(points: NDArray) -> tuple[tuple[tuple[float, float]], float]:
    """Compute the minimum distance between points using an optimized approach. Referenced the pseudocode from [GeeksForGeeks](https://www.geeksforgeeks.org/dsa/closest-pair-of-points-using-divide-and-conquer-algorithm/). 

    Arguments
    ---------
        points: NDArray
            An array of shape (n, d) where n is the number of points and d is the dimension.

    Returns
    -------
        points: tuple[tuple[float, float]]
            An array of shape (2, d), i.e., the pair of points with the minimum distance.
        distance: float
            The minimum distance between the closest pair of points.
    """

    # Steps: 
    # 1. Sort points by x-coordinate
    # 2. Recursively divide the set of points into two halves
    # 3. Find the closest pair in each half
    # 4. Find the closest pair across the dividing line
    # 5. Return the overall closest pair

    # 1. Sort points by x-coordinate
    sorted_by_x = points[points[:, 0].argsort()]

    # 2. Recursively divide the set of points into two halves
    def divide_in_half(pts: NDArray) -> float:
        # Base cases
        if len(pts) == 1:
            return (([0, 0], [0, 0]), float('inf'))
        elif len(pts) == 2:
            return (pts.tolist(), np.linalg.norm(pts[0] - pts[1]))
        
        # Recursive case
        mid = len(pts) // 2
        pts_left, d_left = divide_in_half(pts[:mid])
        pts_right, d_right = divide_in_half(pts[mid:])

        # 3. Find the closest pair in each half
        if d_left < d_right:
            closest_pair = pts_left
            min_distance = d_left
        else:
            closest_pair = pts_right
            min_distance = d_right

        # Build a strip: Collect points whose x-distance from the midline is ≤ d.
        y_coord_strip = pts[np.abs(pts[:, 0] - pts[mid, 0]) <= min_distance]

        # Sort the strip points by y-coordinate
        sorted_by_y = y_coord_strip[y_coord_strip[:, 1].argsort()]

        # 4. Find the closest pair across the dividing line
        # For each point in the strip, compare it with the next up to 7 points (having y-distance ≤ d) to check for closer pairs.
        # NOTE: I (Ina) do not understand why it's 7 points specifically, but this is what the pseudocode says.
        for i in range(len(sorted_by_y)-7):
            for j in range(i+1, i+8):
                if (sorted_by_y[j][1] - sorted_by_y[i][1]) > min_distance:
                    break
                dist = np.linalg.norm(sorted_by_y[i] - sorted_by_y[j])
                if dist < min_distance:
                    min_distance = dist
                    closest_pair = (sorted_by_y[i].tolist(), sorted_by_y[j].tolist())

        # 5. Return the overall closest pair
        return closest_pair, min_distance

    return divide_in_half(sorted_by_x)


def collect_timing_data(output_file: str = "timing_results.xlsx") -> None:
    """Collect timing data for both algorithms and save to Excel file.
    
    Tests both brute force and divide-and-conquer algorithms with input sizes
    n = 1, 10, 50, 100, 250, 500, running 10 trials for each size.
    
    Arguments
    ---------
        output_file: str
            The name of the Excel file to save results to.
    """
    from openpyxl import Workbook # run `pip install openpyxl` in this directory in your console/terminal to get the module
    from openpyxl.styles import Font, PatternFill, Alignment
    
    test_sizes = [1, 100, 200, 300, 400, 500]
    num_trials = 10
    
    # Store results
    brute_force_times = {n: [] for n in test_sizes}
    divide_conquer_times = {n: [] for n in test_sizes}
    
    print("Collecting timing data...")
    
    for n in test_sizes:
        print(f"\nTesting with n={n} points...")
        
        for trial in range(num_trials):
            # Generate random points
            coords = np.random.uniform(-100, 100, n * 2)
            points = coords.reshape((n, 2))
            
            # Time brute force
            start = time()
            brute_force(points)
            bf_time = time() - start
            brute_force_times[n].append(bf_time)
            
            # Time divide and conquer
            start = time()
            divide_and_conquer(points)
            dc_time = time() - start
            divide_conquer_times[n].append(dc_time)
            
            print(f"  Trial {trial + 1}/10: BF={bf_time:.6f}s, D&C={dc_time:.6f}s")
    
    print(f"\nSaving results to {output_file}...")
    
    # Create Excel workbook
    wb = Workbook()
    
    # Sheet 1: Brute Force Data
    bf_sheet = wb.active
    bf_sheet.title = "Brute Force"
    
    # Headers
    bf_sheet['A1'] = 'Input Size (n)'
    bf_sheet['A1'].font = Font(bold=True)
    for i in range(num_trials):
        bf_sheet.cell(1, i + 2, f'Trial {i + 1}')
        bf_sheet.cell(1, i + 2).font = Font(bold=True)
    bf_sheet.cell(1, num_trials + 2, 'Mean')
    bf_sheet.cell(1, num_trials + 2).font = Font(bold=True, color='0000FF')
    bf_sheet.cell(1, num_trials + 3, 'Std Dev')
    bf_sheet.cell(1, num_trials + 3).font = Font(bold=True, color='0000FF')
    
    # Data
    for idx, n in enumerate(test_sizes, start=2):
        bf_sheet.cell(idx, 1, n)
        for trial_idx, time_val in enumerate(brute_force_times[n], start=2):
            bf_sheet.cell(idx, trial_idx, time_val)
        
        # Formulas for mean and std dev
        bf_sheet.cell(idx, num_trials + 2, f'=AVERAGE(B{idx}:{chr(65 + num_trials)}{idx})')
        bf_sheet.cell(idx, num_trials + 3, f'=STDEV.S(B{idx}:{chr(65 + num_trials)}{idx})')
    
    # Formatting
    bf_sheet.column_dimensions['A'].width = 15
    for col in range(2, num_trials + 4):
        bf_sheet.column_dimensions[chr(64 + col)].width = 12
    
    # Sheet 2: Divide and Conquer Data
    dc_sheet = wb.create_sheet("Divide and Conquer")
    
    # Headers
    dc_sheet['A1'] = 'Input Size (n)'
    dc_sheet['A1'].font = Font(bold=True)
    for i in range(num_trials):
        dc_sheet.cell(1, i + 2, f'Trial {i + 1}')
        dc_sheet.cell(1, i + 2).font = Font(bold=True)
    dc_sheet.cell(1, num_trials + 2, 'Mean')
    dc_sheet.cell(1, num_trials + 2).font = Font(bold=True, color='0000FF')
    dc_sheet.cell(1, num_trials + 3, 'Std Dev')
    dc_sheet.cell(1, num_trials + 3).font = Font(bold=True, color='0000FF')
    
    # Data
    for idx, n in enumerate(test_sizes, start=2):
        dc_sheet.cell(idx, 1, n)
        for trial_idx, time_val in enumerate(divide_conquer_times[n], start=2):
            dc_sheet.cell(idx, trial_idx, time_val)
        
        # Formulas for mean and std dev
        dc_sheet.cell(idx, num_trials + 2, f'=AVERAGE(B{idx}:{chr(65 + num_trials)}{idx})')
        dc_sheet.cell(idx, num_trials + 3, f'=STDEV.S(B{idx}:{chr(65 + num_trials)}{idx})')
    
    # Formatting
    dc_sheet.column_dimensions['A'].width = 15
    for col in range(2, num_trials + 4):
        dc_sheet.column_dimensions[chr(64 + col)].width = 12
    
    # Sheet 3: Summary
    summary_sheet = wb.create_sheet("Summary")
    
    # Headers
    summary_sheet['A1'] = 'Input Size (n)'
    summary_sheet['B1'] = 'Brute Force Mean (s)'
    summary_sheet['C1'] = 'Brute Force Std Dev (s)'
    summary_sheet['D1'] = 'D&C Mean (s)'
    summary_sheet['E1'] = 'D&C Std Dev (s)'
    summary_sheet['F1'] = 'Speedup Factor'
    
    for cell in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1']:
        summary_sheet[cell].font = Font(bold=True)
        summary_sheet[cell].fill = PatternFill(start_color='D3D3D3', end_color='D3D3D3', fill_type='solid')
    
    # Data with formulas referencing other sheets
    for idx, n in enumerate(test_sizes, start=2):
        summary_sheet.cell(idx, 1, n)
        summary_sheet.cell(idx, 2, f"='Brute Force'!L{idx}")
        summary_sheet.cell(idx, 3, f"='Brute Force'!M{idx}")
        summary_sheet.cell(idx, 4, f"='Divide and Conquer'!L{idx}")
        summary_sheet.cell(idx, 5, f"='Divide and Conquer'!M{idx}")
        summary_sheet.cell(idx, 6, f"=B{idx}/D{idx}")
    
    # Formatting
    summary_sheet.column_dimensions['A'].width = 15
    for col in ['B', 'C', 'D', 'E', 'F']:
        summary_sheet.column_dimensions[col].width = 20
    
    # Save workbook
    wb.save(output_file)
    print(f"Results saved successfully to {output_file}")
    print("\nSummary:")
    print(f"  Test sizes: {test_sizes}")
    print(f"  Trials per size: {num_trials}")
    print(f"  Total tests run: {len(test_sizes) * num_trials * 2}")


def main(args: argparse.Namespace) -> None: 
    """Test the closest pair algorithms. 
    
    Pass `--num-points [NUMBER] or -np [NUMBER]` to specify the number of points to generate.
    """

    num_points: int = args.num_points

    # Bulk generate sample points using numpy
    coords = np.random.uniform(-100, 100, num_points * 2)  # generate num_points*2 numbers each with value between -100 and 100
    points = coords.reshape((num_points, 2))            # reshape to (num_points, 2)             

    # Test
    brute_start = time()
    print(f"Testing brute-force algorithm with {num_points} points...")
    brute_closest, brute_distance = brute_force(points)
    brute_time = time() - brute_start
    brute_closest_str: str = [f"({brute_closest[i][0]:.2f}, {brute_closest[i][1]:.2f})" for i in range(2)]
    print(f"Brute-force closest points found are {brute_closest_str} with minimum distance {brute_distance:.6f}")
    print(f"Brute-force time taken: {brute_time:.4f} seconds\n")

    dnc_start = time()
    print(f"Testing optimized algorithm with {num_points} points...")
    dnc_closest, dnc_distance = divide_and_conquer(points)
    dnc_time = time() - dnc_start
    dnc_closest_str: str = [f"({dnc_closest[i][0]:.2f}, {dnc_closest[i][1]:.2f})" for i in range(2)]
    print(f"Optimized algorithm closest points found are {dnc_closest_str} with minimum distance {dnc_distance:.6f}")
    print(f"Optimized algorithm time taken: {dnc_time:.4f} seconds\n")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Closest pair of points algorithms.")
    parser.add_argument("--num-points", "-np", type=int, help="number of points", default=5000)
    parser.add_argument("--collect-data", "-cd", action="store_true", help="collect timing data and save to Excel")
    parser.add_argument("--output-file", "-o", type=str, default="timing_results.xlsx", help="output Excel file name")
    args = parser.parse_args()
    
    if args.collect_data:
        collect_timing_data(args.output_file)
    else:
        main(args)