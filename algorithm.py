# usr/bin/env python3
"""
Prerequisite: `pip install numpy` in this directory in your console/terminal if you don't have it already.
Run `python3 algorithm.py` to test the closest pair algorithms implemented below.
"""

__author__ = "Alex Bennet, Ella Kocher, Yina Tang"
__description__ = "Implementation of closest pair algorithms. (Project 1 for COMP 422, Fall 2025)"

import argparse
from time import time
from math import isclose
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
    def divide_in_half(A: NDArray) -> tuple[tuple[float], float]:
        # Base cases
        if len(A) <= 1:
            return (([float('-inf'), float('-inf')], [float('inf'), float('inf')]), float('inf'))
        elif len(A) == 2:
            return (A.tolist(), float(np.linalg.norm(A[0] - A[1])))
        
        # Recursive case
        mid_i = len(A) // 2
        cp_left, d_left = divide_in_half(A[:mid_i])
        cp_right, d_right = divide_in_half(A[mid_i:])

        # 3. Find the closest pair in each half
        if d_left < d_right:
            cp = cp_left
            d = d_left
        else:
            cp = cp_right
            d = d_right

        # Build a strip: Collect points whose x-distance from the midline is ≤ d.
        y_coord_strip = A[np.abs(A[:, 0] - A[mid_i, 0]) <= d]

        # Sort the strip points by y-coordinate
        sorted_by_y = y_coord_strip[y_coord_strip[:, 1].argsort()]

        # 4. Find the closest pair across the dividing line
        # For each point in the strip, compare it with the next up to 7 points (having y-distance ≤ d) to check for closer pairs.
        for i in range(len(sorted_by_y)):
            stop = min(i + 8, len(sorted_by_y))
            for j in range(i+1, stop):
                if (sorted_by_y[j][1] - sorted_by_y[i][1]) > d:
                    break

                dist = np.linalg.norm(sorted_by_y[i] - sorted_by_y[j])
                if dist < d:
                    d = dist
                    cp = (sorted_by_y[i].tolist(), sorted_by_y[j].tolist())

        # 5. Return the overall closest pair
        return cp, float(d)

    return divide_in_half(sorted_by_x)


def main(args: argparse.Namespace) -> None: 
    """Test the closest pair algorithms. 
    
    Pass `--num-points [NUMBER] ... or -np [NUMBER] [NUMBER] ...` to specify the number of points to generate.
    """

    print("+-------------------------------------------+")
    print("| Closest Pair Algorithms Performance Test* |")
    print("+-------------------------------------------+")
    print("*This will run both the brute-force and divide-and-conquer algorithms on random points and compare their execution times.")
    print("*It will also verify that both algorithms return the same pair of points by comparing the minimum distances.\n")
    print("Test sizes (number of points):", args.num_points)

    sizes: list[int] = args.num_points

    bf_results = []
    dc_results = []
    bf_times = []
    dc_times = []

    for size in sizes:
        print(f"Running algorithms for {size} points...")
        points = np.random.rand(size, 2) * 1000  # Generate random points in 2D space

        start_time = time()
        bf_result = brute_force(points)
        bf_results.append(bf_result)
        bf_times.append(time() - start_time)

        start_time = time()
        dc_result = divide_and_conquer(points)
        dc_results.append(dc_result)
        dc_times.append(time() - start_time)

        # print(f"Brute Force Result: {bf_result}, Time Taken: {bf_times[-1]:.6f} seconds")
        # print(f"Divide & Conquer Result: {dnc_result}, Time Taken: {dc_times[-1]:.6f} seconds")

        assert isclose(bf_result[1], dc_result[1]), "Results from both algorithms do not match!"
    
    print("""
+------------------------+------------------------+---------------------------+----------------------+---------------------------+
| Number of Points       | Brute Force Distance   | Divide & Conquer Distance | Brute Force Time (s) | Divide & Conquer Time (s) |
+------------------------+------------------------+---------------------------+----------------------+---------------------------+""")
    for size, bf_result, dc_result, bf_time, dc_time in zip(sizes, bf_results, dc_results, bf_times, dc_times):
        print(f"| {size:<22} | {bf_result[1]:<22.6f} | {dc_result[1]:<25.6f} | {bf_time:<20.6f} | {dc_time:<25.6f} |")
        print("+------------------------+------------------------+---------------------------+----------------------+---------------------------+")



if __name__ == "__main__":
    # Parse command-line arguments for number of points
    parser = argparse.ArgumentParser(description="A simple greeting script.")
    parser.add_argument("--num-points", "-np", type=int, nargs="*", default=[10, 50, 100, 500, 1000, 5000, 10000], 
                        metavar="NUMBER",
                        help="Number of points to generate for testing the closest pair algorithms. Enter multiple values separated by spaces to test different sizes. Default: [10, 50, 100, 500, 1000, 5000, 10000]")
    args = parser.parse_args()
    main(args)