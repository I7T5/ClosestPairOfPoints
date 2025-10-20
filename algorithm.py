__author__ = "Alex Bennet, Ella Kocher, Yina Tang"
__description__ = "Implementation of closest pair algorithms. (Project 1 for COMP 422, Fall 2025)"

import argparse
from time import time
import numpy as np  # run `pip install numpy` if you don't have it


def brute_force(points: np.ndarray) -> tuple[tuple[tuple[float, float]], float]:
    """Compute the minimum distance between points using brute-force approach.

    Arguments
    ---------
        points: np.ndarray
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


def optimized_algorithm(points: np.ndarray) -> tuple[tuple[tuple[float, float]], float]:
    """Compute the minimum distance between points using an optimized approach.

    Arguments
    ---------
        points: np.ndarray
            An array of shape (n, d) where n is the number of points and d is the dimension.

    Returns
    -------
        points: tuple[tuple[float, float]]
            An array of shape (2, d), i.e., the pair of points with the minimum distance.
        distance: float
            The minimum distance between the closest pair of points.
    """
    # Placeholder for an optimized algorithm (e.g., divide and conquer)
    # TODO: remove this placeholder and implement the optimized algorithm
    return brute_force(points)


def main() -> None: 
    """Test the closest pair algorithms. 
    
    Pass `--num-points [NUMBER] or -np [NUMBER]` to specify the number of points to generate.
    """

    # Parse command-line arguments for number of points
    parser = argparse.ArgumentParser(description="A simple greeting script.")
    parser.add_argument("--num-points", "-np", type=int, help="number of points", default="5000")
    args = parser.parse_args()
    num_points: int = args.num_points

    # Bulk generate sample points using numpy
    coords = np.random.uniform(0, 100, num_points * 2)  # generate num_points*2 numbers each with value between 0 and 100
    points = coords.reshape((num_points, 2))            # reshape to (num_points, 2)             

    # Test
    brute_start = time()
    print(f"Testing brute-force algorithm with {num_points} points...")
    closest_brute, distance_brute = brute_force(points)
    brute_time = time() - brute_start
    closest_brute_str: str = [f"({closest_brute[i][0]:.2f}, {closest_brute[i][1]:.2f})" for i in range(2)]
    print(f"Brute-force closest points found are {closest_brute_str} with minimum distance {distance_brute:.6f}")
    print(f"Brute-force time taken: {brute_time:.4f} seconds\n")

    opt_start = time()
    print(f"Testing optimized algorithm with {num_points} points...")
    closest_opt, distance_opt = optimized_algorithm(points)
    opt_time = time() - opt_start
    closest_opt_str: str = [f"({closest_opt[i][0]:.2f}, {closest_opt[i][1]:.2f})" for i in range(2)]
    print(f"Optimized algorithm closest points found are {closest_opt_str} with minimum distance {distance_opt:.6f}")
    print(f"Optimized algorithm time taken: {opt_time:.4f} seconds\n")


if __name__ == "__main__":
    main()