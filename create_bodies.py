import numpy as np
import random

def generate_positions(n, r):
    positions = np.zeros((n, 3))
    r_squared = r**2
    for i in range(n):
        radius_squared = r_squared
        while radius_squared >= r_squared:
            x = random.uniform(-1, 1)*r
            y = random.uniform(-1, 1)*r
            z = random.uniform(-1, 1)*r
            radius_squared = x**2 + y**2 + z**2
        positions[i,0] = x
        positions[i,1] = y
        positions[i,2] = z
    return positions

def write_data(filename, m, positions, n):
    with open(filename, "w+") as f:
        for i in range(n):
            x = positions[i, 0]
            y = positions[i, 1]
            z = positions[i, 2]
            list = [m, x, y, z, 0.0, 0.0, 0.0]
            f.write(','.join(map(str, list)))
            f.write("\n")

def create_file(filename, N, M, R):
    positions = generate_positions(N, R)
    write_data(filename, M, positions, N)


# ~~~~~ Start of program ~~~~~
if __name__ == "__main__":
    N = 10
    mass = 2e29
    radius = 100e15
    create_file("initial_conditions.txt", N, mass, radius)
