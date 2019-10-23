import numpy as np
from matplotlib import pyplot

timestep = 10*24*60*60            # 10 days
G = 6.67408e-11
m_sun = 1.9889e30
m_earth = 5.9742e24
m_jup =  1.8986e27
R = 1.4960e11                           # earth-sun distance
r_sun = R*(m_earth/(m_sun+m_earth))     # distance from sun to CoM
r_earth = R - r_sun                     # distance from earth to CoM
r_jup = 7.7854e11
v_sun = np.sqrt(G*m_earth*r_sun/R**2)          # sun circular orbit velocity
v_earth = np.sqrt(G*m_sun*r_earth/R**2)        # earth circular orbit velocity
v_jup = np.sqrt(G*m_sun/r_jup)
period = np.sqrt(4*np.pi**2*R**3/(G*(m_sun+m_earth)))
period_jup = np.sqrt(4*np.pi**2*r_jup**3/(G*(m_sun+m_jup)))
angle = 2*np.pi*(timestep/2)/period
angle_jup = 2*np.pi*(timestep/2)/period_jup
v_earth_x = v_earth*np.sin(angle)
v_earth_y = -1*v_earth*np.cos(angle)
v_sun_x = -1*v_sun*np.sin(angle)
v_sun_y = v_sun*np.cos(angle)
v_jup_x = -1*v_jup*np.sin(angle_jup)
v_jup_y = v_jup*np.cos(angle_jup)

def initial_conditions():
    masses = np.array([m_sun, m_earth, m_jup])                                 # mass of sun, earth
    initial_positions = np.array([[-1*r_sun, 0, 0], [r_earth, 0, 0], [-1*r_jup, 0, 0]])    # x,y,z positions
    initial_velocities = np.array([[v_sun_x, v_sun_y, 0], [v_earth_x, v_earth_y, 0], [v_jup_x, v_jup_y, 0]])
    return masses, initial_positions, initial_velocities

def simulate_paths((masses, initial_positions, initial_velocities)):
    time_limit = 100*365*24*60*60 # 100 years
    time = 0
    dt = timestep
    num_steps = time_limit/dt
    step = 1            # start at step=1 as step=0 is initial conditions
    num_particles = len(masses)
    positions = np.zeros((num_steps, num_particles, 3)) # [time, particle, dimension]
    velocities = np.zeros((num_steps, num_particles, 3))
    k_energies = np.zeros((num_steps, num_particles))
    p_energies = np.zeros((num_steps, num_particles))
    positions[0] = initial_positions
    velocities[0] = initial_velocities
    forces = np.zeros((num_particles, 3))

    while time < time_limit:
        forces = compute_forces(masses, positions[step-1])
        velocities[step] = velocities[step-1] + (forces/masses[:, None])*dt
        positions[step] = positions[step-1] + velocities[step]*dt
        step += 1
        time = step*dt
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
            r_squared = x_sep**2 + y_sep**2
            force = G*m1*m2/r_squared
            if x_sep !=0:
                if x_sep > 0:
                    if y_sep > 0:
                        theta = np.arctan(y_sep/x_sep)
                    else:
                        theta = np.arctan(y_sep/x_sep) + 2*np.pi
                if x_sep < 0:
                    theta = np.arctan(y_sep/x_sep) + np.pi
            elif y_sep > 0:
                theta = np.pi/2
            else:
                theta = (3/2)*np.pi

            particle_forces[p1][p2][0] = force*np.cos(theta)
            particle_forces[p1][p2][1] = force*np.sin(theta)
        resultant_forces[p1, 0] = np.sum(particle_forces[p1, :, 0])
        resultant_forces[p1, 1] = np.sum(particle_forces[p1, :, 1])
        resultant_forces[p1, 2] = np.sum(particle_forces[p1, :, 2])
    return resultant_forces

def calculate_energies(masses, positions, velocities, time_array):
    k_energies = np.zeros(len(time_array))
    p_energies = np.zeros(len(time_array))
    separations = np.zeros(len(time_array))
    for i in xrange(len(time_array)):
        for body1 in xrange(len(masses)):
            v_x = (velocities[i, body1, 0] + velocities[i+1, body1, 0])/2
            v_y = (velocities[i, body1, 1] + velocities[i+1, body1, 1])/2
            v_z = (velocities[i, body1, 2] + velocities[i+1, body1, 2])/2
            v_squared = v_x**2 + v_y**2 + v_z**2
            k_energies[i] += (masses[body1] * v_squared)/2
            for body2 in xrange(len(masses)):
                if body2 > body1:
                    m1=masses[body1]
                    m2=masses[body2]
                    x_sep = positions[i][body2][0] - positions[i][body1][0]
                    y_sep = positions[i][body2][1] - positions[i][body1][1]
                    r = np.sqrt(x_sep**2 + y_sep**2)
                    p_energies[i] += -G*m1*m2/r
        x_sep = positions[i][0][0] - positions[i][1][0]
        y_sep = positions[i][0][1] - positions[i][1][1]
        separations[i] = np.sqrt(x_sep**2 + y_sep**2)
    return k_energies, p_energies, separations

def output_graphs(positions, k_energy, p_energy, separations, time, short_time):
    pyplot.figure()
    pyplot.subplot(221)
    pyplot.plot(positions[:,0,0], positions[:,0,1], label="Sun")
    pyplot.plot(positions[:,1,0], positions[:,1,1], label="Earth")
    pyplot.plot(positions[:,2,0], positions[:,2,1], label="Jupiter")
    pyplot.xlabel("x coordinate (m)")
    pyplot.ylabel("y coordinate (m)")
    pyplot.legend()

    pyplot.subplot(222)
    pyplot.plot(short_time, k_energy, label="Kinetic energy")
    pyplot.plot(short_time, p_energy, label="Potential energy")
    total_energy = k_energy + p_energy
    pyplot.plot(short_time, total_energy, label="Total energy")
    pyplot.xlabel("Time (s)")
    pyplot.ylabel("Energy (J)")
    pyplot.legend()

    initial_energy = total_energy[0]
    final_energy = total_energy[len(total_energy)-1]
    percent_change = abs(100*(final_energy-initial_energy)/initial_energy)
    print "change in total energy is", str(percent_change) + "%"

    pyplot.subplot(212)
    pyplot.plot(short_time, separations)
    pyplot.xlabel("Time (s)")
    pyplot.ylabel("Earth-Sun distance (m)")
    pyplot.show()

positions, velocities, time_array, short_time_array = simulate_paths(initial_conditions())
k_energy, p_energy, separations = calculate_energies(initial_conditions()[0], positions, velocities, short_time_array)
output_graphs(positions, k_energy, p_energy, separations, time_array, short_time_array)

# v starts at -1/2
# account for case where x_sep or y_sep are zero
