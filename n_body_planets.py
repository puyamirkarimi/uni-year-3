from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import random
#import matplotlib.patches as patches
import time
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

theta_value = 0.5
AU = 149.6e9      # 1 AU in metres
G = 6.67408e-11
epsilon = 0
ep_squared = epsilon**2

class Node:
    children = None     # points to children nodes
    mass = None         # combined mass of particles in octant
    centre_of_mass = None
    box = None          # octant dimensions

def simulate(particles, dt, time_limit, skips):
    start_time = time.time()

    n = len(particles)      # number of particles
    num_steps = time_limit/dt
    positions = np.zeros((int(np.ceil(num_steps/skips)), n, 3)) # [time, particle, dimension]
    velocities = np.zeros((int(np.ceil(num_steps/skips)), n, 3))
    k_energies = np.zeros((int(np.ceil(num_steps/skips)), n))
    p_energies = np.zeros((int(np.ceil(num_steps/skips)), n))
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
        if step % skips == 0:
            positions[int(step/skips)] = particles[:,1:4]
            velocities[int(step/skips)] = particles[:,4:]
        step += 1

    end_time = time.time()
    sim_time = end_time - start_time
    print "simulation time:", sim_time
    return positions, np.arange(0, time_limit-dt, dt*skips)

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
                array.append([float(m),float(x)*AU,float(y)*AU,float(z)*AU,float(vx)*1000,float(vy)*1000,float(vz)*1000])
                #array.append([float(m),float(x),float(y),float(z),float(vx),float(vy),float(vz)])
    return np.array(array)

def output((positions, time_array)):
    plt.figure()
    for body in xrange(len(positions[0,:,0])):
        plt.plot(positions[:,body,0], positions[:,body,1])
        for t in xrange(len(time_array)):
            plt.scatter(positions[t,body,0], positions[t,body,1])
    plt.show()

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

def output2(positions, frames):
    rad = 31
    n = 9     # num of particles
    # Attach 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    data = []
    for i in range(n):
        data.append(make_data(positions, i, frames)/AU)

    lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], 'o')[0] for dat in data]

    # Set the axes properties
    ax.set_xlim3d([-rad, rad])
    ax.set_xlabel('x (AU)')

    ax.set_ylim3d([-rad, rad])
    ax.set_ylabel('y (AU)')

    ax.set_zlim3d([-rad, rad])
    ax.set_zlabel('z (AU)')

    ax.set_title('Solar System: N=9, T=200yr')

    # Creating the Animation object
    ani = animation.FuncAnimation(fig, update_lines, frames, fargs=(data, lines),
                                  interval=10, repeat=False, blit=False)
    ani.save('solar.gif', writer='imagemagick')
    plt.show()


# ~~~~~ Start of program ~~~~~
if __name__ == "__main__":
    filename = ("bodies.txt")
    #filename = ("initial_conditions.txt")
    particles = read_data(filename)
    timestep = 10*24*60*60
    time_limit = 200*365*24*60*60
    skips = 100
    frames = int(time_limit/(timestep*skips))
    #output(simulate(particles, timestep, time_limit, skips))
    output2(simulate(particles, timestep, time_limit, skips)[0], frames)
    #simulate(particles, timestep, time_limit, skips)
