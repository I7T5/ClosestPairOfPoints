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
    # Parse command-line arguments for number of points
    parser = argparse.ArgumentParser(description="A simple greeting script.")
    parser.add_argument("--num-points", "-np", type=int, help="number of points", default="5000")
    args = parser.parse_args()
    main(args)