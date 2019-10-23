import numpy as np
import random

G = 6.67408e-11
pc = 3.086e16               # 1 parsec in metres

'''def generate_positions_velocities(n, a, m, M):
    max_rad = 10*a
    positions = np.zeros((n, 3))
    velocities = np.zeros((n, 3))
    for i in range(n):
        radius_squared = max_rad
        rand = 1
        while radius_squared >= max_rad**2 or rand >= plummer(radius_squared, a):
            #print rand >= plummer(radius_squared, a)
            # REJECTION SAMPLING
            x = random.uniform(-1, 1)*max_rad
            y = random.uniform(-1, 1)*max_rad
            z = random.uniform(-1, 1)*max_rad
            radius_squared = x**2 + y**2 + z**2
            rand = random.uniform(0, 1)
        positions[i,0] = x
        positions[i,1] = y
        positions[i,2] = z

        f = 0.0
        g = 0.1
        while g > f*f*(1.0-f*f)**3.5:
            f = np.random.uniform(0,1)
            g = np.random.uniform(0,0.1)
        velocity = f * np.sqrt((2.0*G*M)/a) * (1.0 + radius_squared/(a**2))**(-1/4)
        theta = np.arccos(np.random.uniform(-1, 1))
        phi = np.random.uniform(0, 2*np.pi)
        velocities[i, 0] = velocity * np.sin(theta) * np.cos(phi)
        velocities[i, 1] = velocity * np.sin(theta) * np.sin(phi)
        velocities[i, 2] = velocity * np.cos(theta)
    positions, velocities = centre(n, positions, velocities, m, M)
    return positions, velocities'''

def generate_positions_velocities(n, a, masses, M):
    positions = np.zeros((n, 3))
    velocities = np.zeros((n, 3))
    for i in range(n):
        radius = a * (1/np.sqrt(np.random.uniform(0,1)**(-2.0/3.0) - 1.0))
        theta = np.arccos(np.random.uniform(-1, 1))
        phi = np.random.uniform(0, 2*np.pi)
        positions[i,0] = radius * np.sin(theta) * np.cos(phi)
        positions[i,1] = radius * np.sin(theta) * np.sin(phi)
        positions[i,2] = radius * np.cos(theta)

        f = 0.0
        g = 0.1
        while g > f*f*(1.0-f*f)**3.5:
            f = np.random.uniform(0,1)
            g = np.random.uniform(0,0.1)
        velocity = f * np.sqrt((2.0*G*M)/a) * (1.0 + (radius**2)/(a**2))**(-1/4)
        theta = np.arccos(np.random.uniform(-1, 1))
        phi = np.random.uniform(0, 2*np.pi)
        velocities[i, 0] = velocity * np.sin(theta) * np.cos(phi)
        velocities[i, 1] = velocity * np.sin(theta) * np.sin(phi)
        velocities[i, 2] = velocity * np.cos(theta)
    positions, velocities = centre(n, positions, velocities, masses, M)
    return positions, velocities

'''def plummer(radius_squared, a):
    return (1+(radius_squared/a**2))**(-5/2)'''

def centre(n, pos, vel, masses, M):
    pos_CoM = np.zeros(3)   # position of CoM
    vel_CoM = np.zeros(3)   # velocity of CoM
    for i in range(n):
        pos_CoM += pos[i, :]*masses[i]
        vel_CoM += vel[i, :]*masses[i]
    pos_CoM = pos_CoM / M
    vel_CoM = vel_CoM / M
    print "CoM position:", pos_CoM
    print "COM velocity:", vel_CoM
    for i in range(n):
        pos[i] -= pos_CoM
        vel[i] -= vel_CoM
    return pos, vel

def write_data(filename, masses, positions, velocities, n):
    with open(filename, "w+") as f:
        for i in range(n):
            x = positions[i, 0]
            y = positions[i, 1]
            z = positions[i, 2]
            vx = velocities[i, 0]
            vy = velocities[i, 1]
            vz = velocities[i, 2]
            list = [masses[i], x, y, z, vx, vy, vz]
            f.write(','.join(map(str, list)))
            f.write("\n")

def create_file(filename, N, M, a, total_mass):
    positions, velocities = generate_positions_velocities(N, a, M, total_mass)
    write_data(filename, M, positions, velocities, N)


# ~~~~~ Start of program ~~~~~
if __name__ == "__main__":
    N = 500
    m_sun = 2e30
    masses = np.zeros(N)
    for i in range(N/2):
        masses[i] = m_sun
    for i in range(N/2, N):
        masses[i] = m_sun/2
    a = 2*pc          # plummer radius
    M = np.sum(masses)
    print M
    create_file("initial_conditions_5.txt", N, masses, a, M)
