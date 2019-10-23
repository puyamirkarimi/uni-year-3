from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import random
#import matplotlib.patches as patches
import time
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

theta_value = 0.5
AU = 149.6e9                # 1 AU in metres
pc = 3.086e16               # 1 parsec in metres
G = 6.67408e-11

class Node:
    children = None         # points to children nodes
    mass = None             # combined mass of particles in octant
    centre_of_mass = None   # centre of mass of particles in octant
    box = None              # octant boundaries

def simulate(particles, dt, time_limit, skips, ep_squared, M):
    start_time = time.time()

    n = len(particles)      # number of particles
    num_steps = time_limit/dt
    positions = np.zeros((int(np.ceil(num_steps/skips)), n, 3)) # [time, particle, dimension]
    velocities = np.zeros((int(np.ceil(num_steps/skips)), n, 3))
    #k_energies = np.zeros((int(np.ceil(num_steps/skips)), n))
    #p_energies = np.zeros((int(np.ceil(num_steps/skips)), n))
    positions[0] = particles[:,1:4]
    velocities[0] = particles[:,4:]

    forces = np.zeros((n, 3))
    step = 1            # start at step=1 as step=0 is initial conditions
    while step < num_steps:
        root = Node()
        root.center_of_mass = []
        root.box = find_node_box(particles)

        for i in xrange(n):
            # insert each particle as a node in the octree
            insert_node(root, particles[i][0], particles[i][1], particles[i][2], particles[i][3])
        for i in xrange(n):
            fx, fy, fz = compute_force(root, particles[i][0], particles[i][1], particles[i][2], particles[i][3])
            '''forces[i] = (total_fx, total_fy, total_fz)
        for i in xrange(n):
            fx, fy, fz = forces[i][0], forces[i][1], forces[i][2]'''
            particles[i][4] += (fx / particles[i][0]) * dt
            particles[i][5] += (fy / particles[i][0]) * dt
            particles[i][6] += (fz / particles[i][0]) * dt

            particles[i][1] += particles[i][4] * dt
            particles[i][2] += particles[i][5] * dt
            particles[i][3] += particles[i][6] * dt


        '''pos_CoM = np.zeros(3)   # position of CoM
        vel_CoM = np.zeros(3)   # velocity of CoM
        for i in xrange(n):
            pos_CoM += particles[i][1:4]*particles[i][0]
            vel_CoM += particles[i][4:7]*particles[i][0]
        pos_CoM = pos_CoM / M
        vel_CoM = vel_CoM / M
        print "CoM position:", pos_CoM
        print "COM velocity:", vel_CoM
        for i in xrange(n):
            particles[i][1] -= pos_CoM[0]
            particles[i][2] -= pos_CoM[1]
            particles[i][3] -= pos_CoM[2]
            particles[i][4] -= vel_CoM[0]
            particles[i][5] -= vel_CoM[1]
            particles[i][6] -= vel_CoM[2]'''

        if step % skips == 0:
            positions[int(step/skips)] = particles[:,1:4]
            velocities[int(step/skips)] = particles[:,4:]
        step += 1

    end_time = time.time()
    sim_time = end_time - start_time
    print "simulation time:", sim_time
    return np.arange(0, time_limit-(dt*skips), dt*skips), positions, velocities

def insert_node(root, m, x, y, z):
    if root.mass is None:
        # if the root is empty, add the first particle
        root.mass = m
        root.center_of_mass = [x, y, z]
        return
    elif root.children is None:
        # if
        root.children = [None,None,None,None,None,None,None,None]
        old_octant = octant_of_particle(root.box, root.center_of_mass[0], root.center_of_mass[1], root.center_of_mass[2])
        if root.children[old_octant] is None:         ################### can't I take out the if statement?
            root.children[old_octant] = Node()
            root.children[old_octant].box = octant_box(root.box, old_octant)
        insert_node(root.children[old_octant], root.mass, root.center_of_mass[0], root.center_of_mass[1], root.center_of_mass[2])
        new_octant = octant_of_particle(root.box, x, y, z)
        # 2
        if root.children[new_octant] is None:
            root.children[new_octant] = Node()
            root.children[new_octant].box = octant_box(root.box, new_octant)
        insert_node(root.children[new_octant], m, x, y, z)
        root.center_of_mass[0] = (root.center_of_mass[0]*root.mass + x*m) / (root.mass + m)
        root.center_of_mass[1] = (root.center_of_mass[1]*root.mass + y*m) / (root.mass + m)
        root.center_of_mass[2] = (root.center_of_mass[2]*root.mass + z*m) / (root.mass + m)
        root.mass = root.mass + m
    else:
        new_octant = octant_of_particle(root.box, x, y, z)
        if root.children[new_octant] is None:
            root.children[new_octant] = Node()
            root.children[new_octant].box = octant_box(root.box, new_octant)
        insert_node(root.children[new_octant], m, x, y, z)
        root.center_of_mass[0] = (root.center_of_mass[0]*root.mass + x*m) / (root.mass + m)
        root.center_of_mass[1] = (root.center_of_mass[1]*root.mass + y*m) / (root.mass + m)
        root.center_of_mass[2] = (root.center_of_mass[2]*root.mass + z*m) / (root.mass + m)
        root.mass = root.mass + m

def octant_of_particle(box, x, y, z):
    """Return the position of the octant of the particle (x,y,z)"""
    x_centre = (box[1] + box[0]) / 2
    y_centre = (box[3] + box[2]) / 2
    z_centre = (box[4] + box[5]) / 2
    if z >= z_centre:
        if y >= y_centre:
            if x <= x_centre:
                return 0
            else:
                return 1
        else:
            if x >= x_centre:
                return 2
            else:
                return 3
    else:
        if y >= y_centre:
            if x <= x_centre:
                return 4
            else:
                return 5
        else:
            if x >= x_centre:
                return 6
            else:
                return 7

def octant_box(box, octant):
    """Return the coordinate of the quadrant
    """
    x = (box[0] + box[1]) / 2
    y = (box[2] + box[3]) / 2
    z = (box[4] + box[5]) / 2
    #Quadrant 0: (xmin, x, y, ymax)
    if octant == 0:
        return box[0], x, y, box[3], z, box[5]
    #Quadrant 1: (x, xmax, y, ymax)
    elif octant == 1:
        return x, box[1], y, box[3], z, box[5]
    #Quadrant 2: (x, xmax, ymin, y)
    elif octant == 3:
        return box[0], x, box[2], y, z, box[5]
    #Quadrant 3: (xmin, x, ymin, y)
    elif octant == 2:
        return x, box[1], box[2], y, z, box[5]
    elif octant == 4:
        return box[0], x, y, box[3], box[4], z
    elif octant == 5:
        return x, box[1], y, box[3], box[4], z
    elif octant == 7:
        return box[0], x, box[2], y, box[4], z
    elif octant == 6:
        return x, box[1], box[2], y, box[4], z

def find_node_box(array):
    """Create a suitable cube boundary box for the input particles"""
    if len(array) == 0 or len(array) == 1:
        return None
    xmin, xmax = array[0][1], array[0][1]
    ymin, ymax = array[0][2], array[0][2]
    zmin, zmax = array[0][3], array[0][3]
    for i in xrange(len(array)):
        if array[i][1] > xmax:
            xmax = array[i][1]
        if array[i][1] < xmin:
            xmin = array[i][1]
        if array[i][2] > ymax:
            ymax = array[i][2]
        if array[i][2] < ymin:
            ymin = array[i][2]
        if array[i][3] > zmax:
            zmax = array[i][3]
        if array[i][3] < zmin:
            zmin = array[i][3]
    if xmax - xmin == ymax - ymin and ymax - ymin == zmax - zmin:
        return xmin, xmax, ymin, ymax, zmin, zmax
    elif xmax - xmin > ymax - ymin and xmax - xmin > zmax - zmin:
        return xmin, xmax, ymin, ymax+(xmax-xmin-ymax+ymin), zmin, zmax+(xmax-xmin-zmax+zmin)
    elif ymax - ymin > xmax - xmin and ymax - ymin > zmax - zmin:
        return xmin, xmax+(ymax-ymin-xmax+xmin), ymin, ymax, zmin, zmax+(ymax-ymin-zmax+zmin)
    else:
        return xmin, xmax+(zmax-zmin-xmax+xmin), ymin, ymax+(zmax-zmin-ymax+ymin), zmin, zmax

def compute_force(root, m, x1, y1, z1):
    if root.mass is None:
        return 0, 0, 0
    if root.center_of_mass[0] == x1 and root.center_of_mass[1] == y1 and root.center_of_mass[2] == z1 and root.mass == m:
        return 0, 0, 0
    d = root.box[1]-root.box[0]
    x2, y2, z2 = root.center_of_mass[0], root.center_of_mass[1], root.center_of_mass[2]
    r_squared = (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2
    if (d**2)/r_squared < theta_value**2 or root.children is None:
    # everything is squared because (I think) it's more efficient than finding the square root of r_squared
        return force(m,x1,y1,z1, root.mass,x2,y2,z2, r_squared)
    else:
        fx = 0.0
        fy = 0.0
        fz = 0.0
        for i in xrange(8):
            if root.children[i] is not None:
                (fx_child, fy_child, fz_child) = compute_force(root.children[i],m,x1,y1,z1)
                fx += fx_child
                fy += fy_child
                fz += fz_child
    return fx, fy, fz

def force(m1,x1,y1,z1, m2,x2,y2,z2, r_squared):
    '''uses spherical polars, not sure if most efficient way'''
    f = G*m1*m2/(r_squared+ep_squared)
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    phi = np.arctan2(dy, dx)
    dxy = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dxy, dz)
    fz = f * np.cos(theta)
    fx = f * np.sin(theta) * np.cos(phi)
    fy = f * np.sin(theta) * np.sin(phi)
    '''
    f = G*m1*m2/r_squared
    dx = x2 - x1
    dy = y2 - y1
    force = G*m1*m2/r_squared
    if dx !=0:
        if dx > 0:
            if dy > 0:
                theta = np.arctan(dy/dx)
            else:
                theta = np.arctan(dy/dx) + 2*np.pi
        if dx < 0:
            theta = np.arctan(dy/dx) + np.pi
    elif dy > 0:
        theta = np.pi/2
    else:
        theta = (3/2)*np.pi

    fx = force*np.cos(theta)
    fy = force*np.sin(theta)'''

    return fx, fy, fz

def read_data(filename):
    array = []
    with open(filename) as f:
        for line in f:
            if line[0] == '#':
                continue
            else:
                m,x,y,z,vx,vy,vz = line.split(',')
                #array.append([float(m),float(x)*AU,float(y)*AU,float(z)*AU,float(vx)*1000,float(vy)*1000,float(vz)*1000])
                array.append([float(m),float(x),float(y),float(z),float(vx),float(vy),float(vz)])
    return np.array(array)

'''
def output1((positions, time_array)):
    plt.figure()
    for body in xrange(len(positions[0,:,0])):
        plt.plot(positions[:,body,0], positions[:,body,1])
        for t in xrange(len(time_array)):
            plt.scatter(positions[t,body,0], positions[t,body,1])
    plt.show()
'''

def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        line.set_data(data[0:2, num-1:num])
        line.set_3d_properties(data[2, num-1:num])
    return lines

def make_data(array, i, frames):
    data = np.empty((3, frames))
    # array[steps, n ,3]
    for step in range(len(array[:,0,0])):
        for dim in range(len(array[0,0,:])):
            data[dim, step] = array[step, i, dim]
    return data

'''def output2(positions, frames, a):
    rad = (2*a)/pc
    n = len(positions[0,:,0])    # num of particles
    print "n:", n
    # Attach 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    data = []
    for i in range(n):
        data.append(make_data(positions, i, frames)/pc)

    lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], 'o')[0] for dat in data]

    # Set the axes properties
    ax.set_xlim3d([-rad, rad])
    ax.set_xlabel('x (pc)')

    ax.set_ylim3d([-rad, rad])
    ax.set_ylabel('y (pc)')

    ax.set_zlim3d([-rad, rad])
    ax.set_zlabel('z (pc)')

    #ax.set_title('n-body')

    # Creating the Animation object
    ani = animation.FuncAnimation(fig, update_lines, frames, fargs=(data, lines),
                                  interval=1, blit=False)
    #ani.save('cluster.gif', writer='imagemagick')

    plt.show()'''

def calculate_energies(masses, time_array, positions, velocities, a):
    k_energies = np.zeros(len(time_array)-1)
    p_energies = np.zeros(len(time_array)-1)
    separations = np.zeros(len(time_array)-1)
    for i in xrange(len(k_energies)):
        for body1 in xrange(len(masses)):
            v_x = (velocities[i, body1, 0] + velocities[i+1, body1, 0])/2
            v_y = (velocities[i, body1, 1] + velocities[i+1, body1, 1])/2
            v_z = (velocities[i, body1, 2] + velocities[i+1, body1, 2])/2
            v_squared = v_x**2 + v_y**2 + v_z**2
            k_energy = (masses[body1] * v_squared)/2
            p_energy = 0
            k_energies[i] += k_energy
            for body2 in xrange(len(masses)):
                if body2 > body1:
                    m1=masses[body1]
                    m2=masses[body2]
                    x_sep = positions[i][body2][0] - positions[i][body1][0]
                    y_sep = positions[i][body2][1] - positions[i][body1][1]
                    z_sep = positions[i][body2][2] - positions[i][body1][2]
                    r_squared = x_sep**2 + y_sep**2 + z_sep**2
                    p_energy += -G*m1*m2/(np.sqrt(r_squared + a**2))
            p_energies[i] += p_energy
    tot_energies = k_energies + p_energies
    return k_energies, p_energies, tot_energies

def plot_energies(time_long, E_k, E_p , E_tot, n_sims_full):
    n_sims = int(n_sims_full/2)
    t = time_long[0:len(time_long)-1]
    steps = len(t)
    time_gyr = t/(1000000000*365*24*60*60)
    av_E_k = np.zeros(steps)
    av_E_p = np.zeros(steps)
    av_E_tot = np.zeros(steps)
    av_E_vir = np.zeros(steps)
    E_k_vals = np.zeros(n_sims)
    E_k_std_error = np.zeros(steps)
    E_p_vals = np.zeros(n_sims)
    E_p_std_error = np.zeros(steps)
    E_tot_vals = np.zeros(n_sims)
    E_tot_std_error = np.zeros(steps)
    E_vir_vals = np.zeros(n_sims)
    E_vir_std_error = np.zeros(steps)
    for step in range(steps):
        for sim in range(n_sims):
            E_k_vals[sim] = E_k[sim][step]
            E_p_vals[sim] = E_p[sim][step]
            E_tot_vals[sim] = E_tot[sim][step]
            E_vir_vals[sim] = E_p_vals[sim] + 2*E_k_vals[sim]
        av_E_k[step] += np.sum(E_k_vals)/n_sims
        av_E_p[step] += np.sum(E_p_vals)/n_sims
        av_E_tot[step] += np.sum(E_tot_vals)/n_sims
        av_E_vir[step] += np.sum(E_vir_vals)/n_sims
        E_k_std_error[step] = np.std(E_k_vals)/np.sqrt(n_sims)
        E_p_std_error[step] = np.std(E_p_vals)/np.sqrt(n_sims)
        E_tot_std_error[step] = np.std(E_tot_vals)/np.sqrt(n_sims)
        E_vir_std_error[step] = np.std(E_vir_vals)/np.sqrt(n_sims)

    delta_E = np.zeros(len(av_E_tot)-1)
    for i in range(1, len(av_E_tot)):
        delta_E[i-1] = (av_E_tot[i]-av_E_tot[i-1])

    fig1 = plt.figure(1)
    frame1=fig1.add_axes((.12,.3,.8,.6))
    frame1.xaxis.set_visible(False)
    plt.scatter(time_gyr, av_E_k, color='blue')
    plt.scatter(time_gyr, av_E_p, color='purple')
    plt.scatter(time_gyr, av_E_tot, color='red')
    plt.scatter(time_gyr, av_E_vir, color='green')
    plt.errorbar(time_gyr, av_E_k, E_k_std_error, linestyle="None", color='blue')
    plt.errorbar(time_gyr, av_E_p, E_p_std_error, linestyle="None", color='purple')
    plt.errorbar(time_gyr, av_E_tot, E_tot_std_error, linestyle="None", color='red')
    plt.errorbar(time_gyr, av_E_vir, E_vir_std_error, linestyle="None", color='green')
    frame1.yaxis.set_ticks(np.array([-2e38, -1.5e38, -1e38, -0.5e38, 0, 0.5e38, 1e38, 1.5e38, 2e38, 2.5e38]))
    frame1.set_ylim([-2.5e38, 2.5e38])
    plt.xlabel("Time (Gyr)")
    plt.xlim(0, time_gyr[-1])
    plt.ylabel("Energy (J)")
    # 2nd plot
    frame2 = fig1.add_axes((.12,.1,.8,.2))
    #frame2.set_ylabel("Energy change since last timestep (J)")
    frame2.set_xlabel("Time (Gyr)")
    #frame2.set_autoscaley_on(False)
    #frame2.set_ylim([-4, 4])
    #frame2.set_autoscalex_on(False)
    frame2.set_xlim([0, time_gyr[-1]])
    #frame2.yaxis.set_ticks(np.array([-2,0,2]))
    plt.plot(time_gyr[1:], delta_E, 'k+', markersize = 8)
    plt.savefig("4.png")

    fig1 = plt.figure(1)
    frame1=fig1.add_axes((.12,.3,.8,.6))
    frame1.xaxis.set_visible(False)
    plt.scatter(time_gyr, av_E_k, color='blue')
    plt.scatter(time_gyr, av_E_p, color='purple')
    plt.scatter(time_gyr, av_E_tot, color='red')
    plt.scatter(time_gyr, av_E_vir, color='green')
    plt.errorbar(time_gyr, av_E_k, 2*E_k_std_error, linestyle="None", color='blue')
    plt.errorbar(time_gyr, av_E_p, 2*E_p_std_error, linestyle="None", color='purple')
    plt.errorbar(time_gyr, av_E_tot, 2*E_tot_std_error, linestyle="None", color='red')
    plt.errorbar(time_gyr, av_E_vir, 2*E_vir_std_error, linestyle="None", color='green')
    frame1.yaxis.set_ticks(np.array([-2e38, -1.5e38, -1e38, -0.5e38, 0, 0.5e38, 1e38, 1.5e38, 2e38, 2.5e38]))
    frame1.set_ylim([-2.5e38, 2.5e38])
    plt.xlabel("Time (Gyr)")
    plt.xlim(0, time_gyr[-1])
    plt.ylabel("Energy (J)")
    # 2nd plot
    frame2 = fig1.add_axes((.12,.1,.8,.2))
    #frame2.set_ylabel("Energy change since last timestep (J)")
    frame2.set_xlabel("Time (Gyr)")
    #frame2.set_autoscaley_on(False)
    #frame2.set_ylim([-4, 4])
    #frame2.set_autoscalex_on(False)
    frame2.set_xlim([0, time_gyr[-1]])
    #frame2.yaxis.set_ticks(np.array([-2,0,2]))
    plt.plot(time_gyr[1:], delta_E, 'k+', markersize = 8)
    plt.savefig("4_1.png")

    fig1 = plt.figure(1)
    frame1=fig1.add_axes((.12,.3,.8,.6))
    frame1.xaxis.set_visible(False)
    plt.scatter(time_gyr, av_E_k, color='blue')
    plt.scatter(time_gyr, av_E_p, color='purple')
    plt.scatter(time_gyr, av_E_tot, color='red')
    plt.scatter(time_gyr, av_E_vir, color='green')
    plt.errorbar(time_gyr, av_E_k, 3*E_k_std_error, linestyle="None", color='blue')
    plt.errorbar(time_gyr, av_E_p, 3*E_p_std_error, linestyle="None", color='purple')
    plt.errorbar(time_gyr, av_E_tot, 3*E_tot_std_error, linestyle="None", color='red')
    plt.errorbar(time_gyr, av_E_vir, 3*E_vir_std_error, linestyle="None", color='green')
    frame1.yaxis.set_ticks(np.array([-2e38, -1.5e38, -1e38, -0.5e38, 0, 0.5e38, 1e38, 1.5e38, 2e38, 2.5e38]))
    frame1.set_ylim([-2.5e38, 2.5e38])
    plt.xlabel("Time (Gyr)")
    plt.xlim(0, time_gyr[-1])
    plt.ylabel("Energy (J)")
    # 2nd plot
    frame2 = fig1.add_axes((.12,.1,.8,.2))
    #frame2.set_ylabel("Energy change since last timestep (J)")
    frame2.set_xlabel("Time (Gyr)")
    #frame2.set_autoscaley_on(False)
    #frame2.set_ylim([-4, 4])
    #frame2.set_autoscalex_on(False)
    frame2.set_xlim([0, time_gyr[-1]])
    #frame2.yaxis.set_ticks(np.array([-2,0,2]))
    plt.plot(time_gyr[1:], delta_E, 'k+', markersize = 8)
    plt.savefig("4_2.png")

    '''initial_energy = av_E_tot[0]
    final_energy = av_E_tot[-1]
    percent_change = abs(100*(final_energy-initial_energy)/initial_energy)
    print "change in total energy is", str(percent_change) + "%"
    initial_k = av_E_k[0]
    final_k = av_E_k[-1]
    initial_p = av_E_p[0]
    final_p = av_E_p[-1]
    print "U/K start k_energy + pt:", initial_p/initial_k
    print "U/K end:", final_p/final_k'''
    #print "yeets", str(yeets)

def calculate_distribution(pos, M, a, masses):
    sections = 15
    dist = np.zeros(sections)
    #max_r = 1000*a
    section_starts = np.zeros(sections)
    section_width = np.zeros(sections)
    section_width[0] = a/10
    pointer = section_width[0]
    for section in range(1, sections):
        section_width[section] = section_width[section-1] * 1.5
        section_starts[section] = pointer
        pointer = pointer + section_width[section]
    for body in range(1, len(pos[:,0])):
        x0 = pos[body, 0]
        y0 = pos[body, 1]
        z0 = pos[body, 2]
        r_start = np.sqrt(x0**2 + y0**2 + z0**2)
        slot0 = sections-1
        for section in range(sections-1):
            if r_start < section_starts[section+1]:
                slot0 = section
                break
        dist[slot0] += masses[body]
    density = np.zeros(sections)
    for section in range(sections-1):
        vol_curr = (4/3)*np.pi*section_starts[section+1]**3
        vol_prev = (4/3)*np.pi*section_starts[section]**3
        vol = vol_curr - vol_prev
        density[section] = dist[section]/vol
    density[sections-1] = dist[sections-1]/vol
    section_mids = np.zeros(sections)
    for i in range(sections):
        section_mids[i] = section_starts[i] + section_width[i]/2
    return section_mids, density

def plot_distributions(positions, M, a, masses, step_nums, n_sims):
    sections = 15
    av_densities = np.zeros((len(step_nums), sections))
    av_densities2 = np.zeros((len(step_nums), sections))
    std_error = np.zeros((len(step_nums), sections))
    std_error2 = np.zeros((len(step_nums), sections))
    for i in range(len(step_nums)):
        densities = np.zeros((int(n_sims/2), sections))
        for sim in range(int(n_sims/2)):
            section_mids, density = calculate_distribution(centre(positions[sim][step_nums[i],:,:], masses[sim]), M, a, masses[sim])
            densities[sim, :] = density
            av_densities[i, :]+=density/(n_sims/2)
        for step in range(sections):
            std_error[i, step] = np.std(densities[:,step])/np.sqrt(n_sims/2)
        densities = np.zeros((int(n_sims/2), sections))
        for sim in range(int(n_sims/2)):
            section_mids, density2 = calculate_distribution(centre(positions[sim+int(n_sims/2)][step_nums[i],:,:], masses[sim+int(n_sims/2)]), M, a, masses[sim+int(n_sims/2)])
            densities[sim, :] = density2
            av_densities2[i, :] += density2/(n_sims/2)
        for step in range(sections):
            std_error2[i, step] = np.std(densities[:,step])/np.sqrt(n_sims/2)

    x_start = section_mids[1]
    x_end = section_mids[-2]
    x = np.linspace(x_start, x_end, 10000)
    y = plum_dist(x, M, a)
    y2 = plum_dist(x, (M-(200*2e30)), a)

    M1 = 375*2e30
    M2 = M1 - 200*2e30
    C1 = 1/((3*M1)/(4*np.pi*a**3))
    C2 = 1/((3*M2)/(4*np.pi*a**3))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ['red', 'orange']
    colors2 = ['blue', 'green']
    for i in range(len(step_nums)):
        ax.plot(section_mids[1:-1]/a, av_densities[i][1:-1]*C1, color=colors[i])
        ax.scatter(section_mids[1:-1]/a, av_densities[i][1:-1]*C1, color=colors[i])
        ax.errorbar(section_mids[1:-1]/a, av_densities[i][1:-1]*C1, std_error[i,1:-1]*C1, linestyle="None", color=colors[i])
        ax.plot(section_mids[1:-1]/a, av_densities2[i][1:-1]*C2, color=colors2[i])
        ax.scatter(section_mids[1:-1]/a, av_densities2[i][1:-1]*C2, color=colors2[i])
        ax.errorbar(section_mids[1:-1]/a, av_densities2[i][1:-1]*C2, std_error2[i,1:-1]*C2, linestyle="None", color=colors2[i])
    #ax.plot(x/a, y, color="red")
    #ax.plot(x/a, y2, color="blue")
    ax.set_xscale('log')
    ax.set_xlim([(x_start/a)*0.9,x_end/a])
    #ax.set_xlabel("$\rho(r) (kg/m^3)$")
    #ax.set_ylabel("$r/a$")
    ax.set_ylim(bottom=0)
    plt.savefig("distributions.png")
"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ['red', 'orange']
    colors2 = ['blue', 'green']
    for i in range(len(step_nums)):
        ax.plot(section_mids[1:-1]/a, av_densities[i][1:-1], color=colors[i])
        ax.scatter(section_mids[1:-1]/a, av_densities[i][1:-1], color=colors[i])
        ax.errorbar(section_mids[1:-1]/a, av_densities[i][1:-1], std_error[i,1:-1]*3, linestyle="None", color=colors[i])
        ax.plot(section_mids[1:-1]/a, av_densities2[i][1:-1], color=colors2[i])
        ax.scatter(section_mids[1:-1]/a, av_densities2[i][1:-1], color=colors2[i])
        ax.errorbar(section_mids[1:-1]/a, av_densities2[i][1:-1], std_error2[i,1:-1]*3, linestyle="None", color=colors2[i])
    #ax.plot(x/a, y)
    ax.set_xscale('log')
    ax.set_xlim([(x_start/a)*0.9,x_end/a])
    '''ax.set_xlabel("rho(r) (kg/m^3)")
    ax.set_ylabel("r/a")'''
    ax.set_ylim([0, 5e-18])
    plt.savefig("6.png")"""


def plum_dist(x, M, a):
    return ((3*M)/(4*np.pi*a**3))*(1+x**2/a**2)**(-5/2)

def centre(positions, masses):
    pos_CoM = np.zeros(3)   # position of CoM
    for i in xrange(len(positions[:,0])):
        pos_CoM += positions[i, 0:3]*masses[i]
    pos_CoM = pos_CoM / M
    #print "CoM position:", pos_CoM
    for i in xrange(len(positions[:,0])):
        positions[i][0] -= pos_CoM[0]
        positions[i][1] -= pos_CoM[1]
        positions[i][2] -= pos_CoM[2]
    return positions

def CoM_pos(positions, masses):
    pos_CoM = np.zeros(3)   # position of CoM
    for i in xrange(len(positions[:,0])):
        pos_CoM += positions[i, 0:3]*masses[i]
    pos_CoM = pos_CoM / M
    return pos_CoM

def calculate_yonks(positions, time, a, n_sims):
    yonked_bodies = np.zeros((len(positions[0][0,:,0]), int(n_sims/2)))
    av_yonks1 = np.zeros(len(time))
    av_yonks2 = np.zeros(len(time))
    std_error1 = np.zeros(len(time))
    std_error2 = np.zeros(len(time))
    for t in xrange(len(time)):
        ynk = np.zeros(int(n_sims/2))
        for sim in range(int(n_sims/2)):
            for body in xrange(len(positions[0][0,:,0])):
                x = positions[sim][t,body,0]
                y = positions[sim][t,body,1]
                z = positions[sim][t,body,2]
                r_squared = x**2 + y**2 + z**2
                if r_squared > (a*5)**2:
                    yonked_bodies[body, sim] = 1
            ynk[sim] = np.sum(yonked_bodies[:, sim])
            av_yonks1[t] += ynk[sim]/(n_sims/2)
        std_error1[t] = np.std(ynk)/np.sqrt(n_sims/2)
    yonked_bodies = np.zeros((len(positions[0][0,:,0]), int(n_sims/2)))
    for t in xrange(len(time)):
        ynk = np.zeros(int(n_sims/2))
        for sim in range(int(n_sims/2)):
            for body in xrange(len(positions[0][0,:,0])):
                x = positions[sim+int(n_sims/2)][t,body,0]
                y = positions[sim+int(n_sims/2)][t,body,1]
                z = positions[sim+int(n_sims/2)][t,body,2]
                r_squared = x**2 + y**2 + z**2
                if r_squared > (a*5)**2:
                    yonked_bodies[body, sim] = 1
            ynk[sim] = np.sum(yonked_bodies[:, sim])
            av_yonks2[t] += ynk[sim]/(n_sims/2)
        std_error2[t] = np.std(ynk)/np.sqrt(n_sims/2)
    return av_yonks1, av_yonks2, std_error1, std_error2

def plot_yonks(positions, t, a, n_sims):
    yonks1, yonks2, error1, error2 = calculate_yonks(positions, t, a, n_sims)
    t_gyr = t/(1000000000*365*24*60*60)
    plt.figure()
    plt.plot(t/(1000000000*365*24*60*60), yonks1, color='black')
    plt.plot(t/(1000000000*365*24*60*60), yonks2, color='black')
    plt.errorbar(t/(1000000000*365*24*60*60), yonks1, error1, linestyle="None", color='blue')
    plt.errorbar(t/(1000000000*365*24*60*60), yonks2, error2, linestyle="None", color='red')
    plt.ylabel("Number of bodies that have been outside 10pc sphere")
    plt.xlim([t_gyr[0], t_gyr[-1]])
    plt.xlabel("Time (Gyr)")
    plt.savefig("1.png")

    plt.figure()
    plt.plot(t/(1000000000*365*24*60*60), yonks1, color='blue')
    plt.plot(t/(1000000000*365*24*60*60), yonks2, color='red')
    #plt.errorbar(t/(1000000000*365*24*60*60), yonks1, error1, linestyle="None", color='blue')
    #plt.errorbar(t/(1000000000*365*24*60*60), yonks2, error2, linestyle="None", color='red')
    plt.ylabel("Number of bodies that have been outside 10pc sphere")
    plt.xlim([t_gyr[0], t_gyr[-1]])
    plt.xlabel("Time (Gyr)")
    plt.savefig("2.png")

    plt.figure()
    plt.plot(t/(1000000000*365*24*60*60), yonks1, color='blue')
    plt.plot(t/(1000000000*365*24*60*60), yonks2, color='red')
    plt.errorbar(t/(1000000000*365*24*60*60), yonks1, error1, linestyle="None", color='blue')
    plt.errorbar(t/(1000000000*365*24*60*60), yonks2, error2, linestyle="None", color='red')
    plt.ylabel("Number of bodies that have been outside 10pc sphere")
    plt.xlim([t_gyr[0], t_gyr[-1]])
    plt.xlabel("Time (Gyr)")
    plt.savefig("3.png")

def segregation(positions, n_sims, a):
    print "printing average rad, error for no bh start, then bh start, then no bh end, then bh end:"
    av_radii_small = np.zeros(int(n_sims/2))
    av_radii_big = np.zeros(int(n_sims/2))
    radii_small = np.zeros((int(n_sims/2), 250))
    radii_big = np.zeros((int(n_sims/2), 250))
    for sim in range(int(n_sims/2)):
        for body in xrange(250):
            x = positions[sim][0,body,0]
            y = positions[sim][0,body,1]
            z = positions[sim][0,body,2]
            x2 = positions[sim][0,body+250,0]
            y2 = positions[sim][0,body+250,1]
            z2 = positions[sim][0,body+250,2]
            rad = np.sqrt(x**2 + y**2 + z**2)
            rad2 = np.sqrt(x2**2 + y2**2 + z2**2)
            radii_big[sim, body] = rad
            radii_small[sim, body] = rad2
        av_radii_small[sim] = np.median(radii_small[sim, :])
        av_radii_big[sim] = np.median(radii_big[sim, :])
    error_small = np.std(av_radii_small)/np.sqrt(n_sims/2)
    av_radius_small = np.mean(av_radii_small)
    print av_radius_small/a, error_small/a
    error_big = np.std(av_radii_big)/np.sqrt(n_sims/2)
    av_radius_big = np.mean(av_radii_big)
    print av_radius_big/a, error_big/a

    av_radii_small = np.zeros(int(n_sims/2))
    av_radii_big = np.zeros(int(n_sims/2))
    radii_small = np.zeros((int(n_sims/2), 250))
    radii_big = np.zeros((int(n_sims/2), 249))
    for sim in range(int(n_sims/2)):
        for body in xrange(249):
            x = positions[sim+int(n_sims/2)][0,body+1,0]
            y = positions[sim+int(n_sims/2)][0,body+1,1]
            z = positions[sim+int(n_sims/2)][0,body+1,2]
            rad = np.sqrt(x**2 + y**2 + z**2)
            radii_big[sim, body] = rad
        for body in xrange(250):
            x2 = positions[sim+int(n_sims/2)][0,body+250,0]
            y2 = positions[sim+int(n_sims/2)][0,body+250,1]
            z2 = positions[sim+int(n_sims/2)][0,body+250,2]
            rad2 = np.sqrt(x2**2 + y2**2 + z2**2)
            radii_small[sim, body] = rad2
        av_radii_small[sim] = np.median(radii_small[sim, :])
        av_radii_big[sim] = np.median(radii_big[sim, :])
    error_small = np.std(av_radii_small)/np.sqrt(249)
    av_radius_small = np.mean(av_radii_small)
    print av_radius_small/a, error_small/a
    error_big = np.std(av_radii_big)/np.sqrt(n_sims/2)
    av_radius_big = np.mean(av_radii_big)
    print av_radius_big/a, error_big/a

    av_radii_small = np.zeros(int(n_sims/2))
    av_radii_big = np.zeros(int(n_sims/2))
    radii_small = np.zeros((int(n_sims/2), 250))
    radii_big = np.zeros((int(n_sims/2), 250))
    for sim in range(int(n_sims/2)):
        for body in xrange(250):
            x = positions[sim][-1,body,0]
            y = positions[sim][-1,body,1]
            z = positions[sim][-1,body,2]
            x2 = positions[sim][-1,body+250,0]
            y2 = positions[sim][-1,body+250,1]
            z2 = positions[sim][-1,body+250,2]
            rad = np.sqrt(x**2 + y**2 + z**2)
            rad2 = np.sqrt(x2**2 + y2**2 + z2**2)
            radii_big[sim, body] = rad
            radii_small[sim, body] = rad2
        av_radii_small[sim] = np.median(radii_small[sim, :])
        av_radii_big[sim] = np.median(radii_big[sim, :])
    error_small = np.std(av_radii_small)/np.sqrt(n_sims/2)
    av_radius_small = np.mean(av_radii_small)
    print av_radius_small/a, error_small/a
    error_big = np.std(av_radii_big)/np.sqrt(n_sims/2)
    av_radius_big = np.mean(av_radii_big)
    print av_radius_big/a, error_big/a

    av_radii_small = np.zeros(int(n_sims/2))
    av_radii_big = np.zeros(int(n_sims/2))
    radii_small = np.zeros((int(n_sims/2), 250))
    radii_big = np.zeros((int(n_sims/2), 249))
    for sim in range(int(n_sims/2)):
        for body in xrange(249):
            x = positions[sim+int(n_sims/2)][-1,body+1,0]
            y = positions[sim+int(n_sims/2)][-1,body+1,1]
            z = positions[sim+int(n_sims/2)][-1,body+1,2]
            rad = np.sqrt(x**2 + y**2 + z**2)
            radii_big[sim, body] = rad
        for body in xrange(250):
            x2 = positions[sim+int(n_sims/2)][-1,body+250,0]
            y2 = positions[sim+int(n_sims/2)][-1,body+250,1]
            z2 = positions[sim+int(n_sims/2)][-1,body+250,2]
            rad2 = np.sqrt(x2**2 + y2**2 + z2**2)
            radii_small[sim, body] = rad2
        av_radii_small[sim] = np.median(radii_small[sim, :])
        av_radii_big[sim] = np.median(radii_big[sim, :])
    error_small = np.std(av_radii_small)/np.sqrt(249)
    av_radius_small = np.mean(av_radii_small)
    print av_radius_small/a, error_small/a
    error_big = np.std(av_radii_big)/np.sqrt(n_sims/2)
    av_radius_big = np.mean(av_radii_big)
    print av_radius_big/a, error_big/a

def print_CoMs(positions, masses, n_sims, a):
    CoMs1 = np.zeros((int(n_sims/2),3))
    CoMs2 = np.zeros((int(n_sims/2),3))
    r1 = np.zeros(int(n_sims/2))
    r2 = np.zeros(int(n_sims/2))
    for sim in range(int(n_sims/2)):
        CoMs1[sim,:] = CoM_pos(positions[sim][-1,:,:], masses[sim])
        CoMs2[sim,:] = CoM_pos(positions[sim+int(n_sims/2)][-1,:,:], masses[sim+int(n_sims/2)])
        r1[sim] = np.sqrt(CoMs1[sim,0]**2 + CoMs1[sim,1]**2 + CoMs1[sim,2]**2)
        r2[sim] = np.sqrt(CoMs2[sim,0]**2 + CoMs2[sim,1]**2 + CoMs2[sim,2]**2)
    av_CoM1 = np.mean(r1)
    av_CoM2 = np.mean(r2)
    error1 = np.std(r1)/(n_sims/2)
    error2 = np.std(r2)/(n_sims/2)
    print "printing average final centre of mass, error for no black hole and then black hole (units of a)"
    print av_CoM1/a, error1/a
    print av_CoM2/a, error2/a


# ~~~~~ Start of program ~~~~~
if __name__ == "__main__":
    #filename = ("bodies.txt")
    a = 2*pc
    n = 500
    epsilon = a*0.58*(n)**(-0.26)
    ep_squared = epsilon**2
    steps = 90#8000      #90 for thing
    skips = 1#80           # choose such that steps/skips=100
    frames = int(steps/skips)
    n_sims = 10
    k_energies = []
    p_energies = []
    tot_energies = []
    masses = []
    positions = []
    velocities = []
    for i in range(n_sims):
        print i+1
        filename = ("initial_conditions_"+str(i+1)+".txt")
        particles = read_data(filename)
        masses.append(particles[:,0])
        M = np.sum(masses[0])
        timestep = (epsilon/(0.7 * np.sqrt((G * M)/a)))/5
        time_limit = steps*timestep
        t, pos, vel = simulate(particles, timestep, time_limit, skips, ep_squared, M)
        positions.append(pos)
        velocities.append(vel)
        k_E, p_E, tot_E = calculate_energies(masses[i], t, pos, vel, a)
        k_energies.append(k_E)
        p_energies.append(p_E)
        tot_energies.append(tot_E)
    #plot_energies(t, k_energies, p_energies, tot_energies, n_sims)
    #plot_distributions(positions, M, a, masses, [0, (steps/skips)-1], n_sims)
    #plot_yonks(positions, t, a, n_sims)
    #segregation(positions, n_sims, a)
    #print_CoMs(positions, masses, n_sims, a)
    #print "dt:", str(timestep/(1000000*365*24*60*60)) + "Myr"
    #print "epsilon:", epsilon
    plot_distributions(positions, M, a, masses, [0, -1], n_sims)
