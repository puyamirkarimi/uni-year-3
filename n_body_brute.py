import numpy as np
from matplotlib import pyplot as plt
import time

timestep = 100000*365*24*60*60
time_limit = int(2*timestep)
G = 6.67408e-11
AU = 149.6e9      # 1 AU in metres
epsilon = 900000000000000
ep_squared = epsilon**2

def initial_conditions():
    filename = ("initial_conditions.txt")
    masses = []
    initial_positions = []
    initial_velocities = []
    with open(filename) as f:
        for line in f:
            if line[0] == '#':
                continue
            else:
                m,x,y,z,vx,vy,vz = line.split(',')
                masses.append(float(m))
                #initial_positions.append([float(x)*AU,float(y)*AU,float(z)*AU])
                initial_positions.append([float(x),float(y),float(z)])
                #initial_velocities.append([float(vx)*1000,float(vy)*1000,float(vz)*1000])
                initial_velocities.append([float(vx),float(vy),float(vz)])
    return np.array(masses), np.array(initial_positions), np.array(initial_velocities)

def simulate_paths((masses, initial_positions, initial_velocities), dt, time_limit):
    start_time = time.time()

    num_steps = time_limit/dt
    step = 1            # start at step=1 as step=0 is initial conditions
    t = step*dt
    num_particles = len(masses)
    positions = np.zeros((num_steps, num_particles, 3)) # [time, particle, dimension]
    velocities = np.zeros((num_steps, num_particles, 3))
    positions[0] = initial_positions
    velocities[0] = initial_velocities
    forces = np.zeros((num_particles, 3))

    while t < time_limit:
        forces = compute_forces(masses, positions[step-1])
        #print step/num_steps
        velocities[step] = velocities[step-1] + (forces/masses[:, None])*dt
        positions[step] = positions[step-1] + velocities[step]*dt
        step += 1
        t = step*dt

    end_time = time.time()
    sim_time = end_time - start_time
    print "simulation time (brute force):", sim_time
    return positions, velocities, np.arange(0, time_limit, dt), np.arange(0, time_limit-dt, dt)

def compute_forces(masses, pos):
    num_particles = len(masses)
    particle_forces = np.zeros((num_particles, num_particles, 3))
    resultant_forces = np.zeros((num_particles, 3))
    for p1 in xrange(num_particles):
        m1 = masses[p1]
        for p2 in xrange(p1):
            particle_forces[p1][p2] = -1 * particle_forces[p2][p1]
        for p2 in xrange(p1 + 1, num_particles):
            m2 = masses[p2]
            x_sep = pos[p2][0] - pos[p1][0]
            y_sep = pos[p2][1] - pos[p1][1]
            z_sep = pos[p2][2] - pos[p1][2]
            r_squared = x_sep**2 + y_sep**2 + z_sep**2
            f = G*float(m1)*float(m2)/float(r_squared+ep_squared)
            phi = np.arctan2(y_sep, x_sep)
            dxy = np.sqrt(x_sep**2 + y_sep**2)
            theta = np.arctan2(dxy, z_sep)
            particle_forces[p1][p2][2] = f * np.cos(theta)
            particle_forces[p1][p2][0] = f * np.sin(theta) * np.cos(phi)
            particle_forces[p1][p2][1] = f * np.sin(theta) * np.sin(phi)
        resultant_forces[p1, 0] = np.sum(particle_forces[p1, :, 0])
        resultant_forces[p1, 1] = np.sum(particle_forces[p1, :, 1])
        resultant_forces[p1, 2] = np.sum(particle_forces[p1, :, 2])
    return resultant_forces

def output_graphs(positions):
    plt.figure()
    bodies = len(positions[0,:,0])
    total_time = len(positions[:,0,0])
    for body in xrange(bodies):
        #print str(body)+"/"+str(bodies-1)
        plt.plot(positions[:,body,0], positions[:,body,1])
        for t in xrange(total_time):
            plt.scatter(positions[t,body,0], positions[t,body,1])
    plt.show()

positions, velocities, time_array, short_time_array = simulate_paths(initial_conditions(), timestep, time_limit)
#output_graphs(positions)

# v starts at -1/2
# account for case where x_sep or y_sep are zero
