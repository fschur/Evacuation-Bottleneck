import numpy as np
import sys

#----------------------------------------------------------- Room.py ----------------------------------------------------------------

class Room:
    def __init__(self, room, room_size):
        self.room_size = room_size
        if room == "square":
            self.wallshere = False
            self.door_size = room_size/15                    # size of the door is proportional to the size of the room
            self.destination = np.array([[-0.5, room_size/2]])   # destination the agenst want to go
            self.num_walls = 5
            self.walls = np.array([[[0, 0], [0, room_size/2-self.door_size/2]], # wall 1
                          [[0, room_size/2+self.door_size/2], [0, room_size]],  # wall 2
                          [[0, room_size], [room_size, room_size]],             # wall 3
                          [[room_size, room_size], [room_size, 0]],             # wall 4
                          [[room_size, 0], [0, 0]]])                            # wall 5
            # agents spawn with x and y position between 1 and (room_size-1) 
            self.spawn_zone = np.array([[room_size/2, room_size-1], [1, room_size-1]])

        if room == "long_room":
            self.wallshere = False
            self.door_size = room_size/15                    # size of the door is proportional to the size of the room
            self.room_len = room_size
            self.room_with = room_size/5
            self.destination = np.array([[-0.5, self.room_with/2]])   # destination the agenst want to go
            self.num_walls = 5
            self.walls = np.array([[[0, 0], [0, self.room_with/2-self.door_size/2]],        # wall 1
                          [[0, self.room_with/2+self.door_size/2], [0, self.room_with]],    # wall 2
                          [[0, self.room_with], [self.room_len, self.room_with]],           # wall 3
                          [[self.room_len, self.room_with], [self.room_len, 0]],            # wall 4
                          [[self.room_len, 0], [0, 0]]])                                    # wall 5
            # agents spawn with x and y position between 1 and (room_size-1) 
            self.spawn_zone = np.array([[1, self.room_len-1], [1, self.room_with-1]])

        if room == "long_room_v2":
            self.wallshere = False
            self.door_size = room_size/15                    # size of the door is proportional to the size of the room
            self.room_len = room_size
            self.room_with = room_size/5
            self.destination = np.array([[-0.5, self.room_with/2], [self.room_len + 0.5, self.room_with/2]])   # destination the agenst want to go
            self.num_walls = 6
            self.walls = np.array([[[0, 0], [0, self.room_with/2-self.door_size/2]], # wall 1
                          [[0, self.room_with/2+self.door_size/2], [0, self.room_with]],  # wall 2
                          [[0, self.room_with], [self.room_len, self.room_with]],             # wall 3
                          [[self.room_len, self.room_with], [self.room_len, self.room_with/2+self.door_size/2]],
                          [[self.room_len, self.room_with/2-self.door_size/2], [self.room_len, 0]],                 # wall 4
                          [[self.room_len, 0], [0, 0]]])                            # wall 5
            # agents spawn with x and y position between 1 and (room_size-1)
            self.spawn_zone = np.array([[1, self.room_len-1], [1, self.room_with-1]])

        if room == "edu_1":
            self.wallshere = True
            self.door_size = room_size/15                    # size of the door is proportional to the size of the room
            self.destination = np.array([[-0.5, room_size/2]])   # destination the agenst want to go
            self.num_walls = 6
            self.walls = np.array([[[0, 0], [0, room_size/2-self.door_size/2]], # wall 1
                          [[0, room_size/2+self.door_size/2], [0, room_size]],  # wall 2
                          [[0, room_size], [room_size, room_size]],             # wall 3
                          [[room_size, room_size], [room_size, 0]],             # wall 4
                          [[room_size, 0], [0, 0]],                             # wall 5
                          [[room_size/4, room_size*0.3], [room_size/4, room_size*0.7]]])           # wall 6      
            # agents spawn with x and y position between 1 and (room_size-1) 
            self.spawn_zone = np.array([[room_size/2, room_size-1], [1, room_size-1]])

        if room == "edu_room":
            self.wallshere = True
            self.door_size = room_size/15                    # size of the door is proportional to the size of the room
            self.destination = np.array([[-0.5, room_size/2]])   # destination the agenst want to go
            self.num_walls = 7
            self.walls = np.array([[[0, 0], [0, room_size/2-self.door_size/2]], # wall 1
                          [[0, room_size/2+self.door_size/2], [0, room_size]],  # wall 2
                          [[0, room_size], [room_size, room_size]],             # wall 3
                          [[room_size, room_size], [room_size, 0]],             # wall 4
                          [[room_size, 0], [0, 0]],                             # wall 5
                          [[9/25*room_size, room_size*0.15], [5.5/25*room_size, room_size*0.4]],            # wall 6
                          [[5.5/25*room_size, room_size*0.6], [9/25*room_size, room_size*0.85]]])           # wall 7                 
            # agents spawn with x and y position between 1 and (room_size-1) 
            self.spawn_zone = np.array([[room_size/2, room_size-1], [1, room_size-1]])



    def get_wall(self, n):              # gives back the endpoints of the nth wall
        return self.walls[n,:,:]

    def get_num_walls(self):            # gives back the number of walls
        return self.num_walls

    def get_spawn_zone(self):            # gives back the spawn_zone
        return self.spawn_zone

    def get_room_size(self):            # gives back the size of the room
        return self.room_size

    def get_destination(self):          # gives back the destination the agents want to get to
        return self.destination

# --------------------------------------------------------- Integrators.py ----------------------------------------------------------

def exp_midpoint(y0, v0, f, N_steps, dt, room):
    tmp = 0
    agents_escaped = np.zeros(N_steps)
    
    y = np.zeros((y0.shape[0], y0.shape[1], N_steps))
    v = np.zeros((y0.shape[0], y0.shape[1], N_steps))
    a = np.zeros((y0.shape[0], y0.shape[1], N_steps))
    
    y[:,:,0] = y0
    v[:,:,0] = v0

    for k in range(N_steps-1):
        a[:,:,k] = f(y[:,:,k], v[:,:,k])
        v[:,:,k+1] = (v[:,:,k] + dt*f(y[:,:,k] + 0.5*dt*v[:,:,k], v[:,:,k] 
          + 0.5*dt*a[:,:,k]))
        y[:,:,k+1] = y[:,:,k] + dt*v[:,:,k+1]

        for i in range(y.shape[1]):
            # checks if there are two destination and calculates the distance to the closets destination
            destination = np.zeros(len(room.get_destination()))
            for count, des in enumerate(room.get_destination()):
                destination[count] = np.linalg.norm(y[:, i, k + 1] - des)
            distance = np.amin(destination)

            if distance < 0.5:
                #we have to use position of door here instead of (0,5)
                y[:,i,k+1] = 10**6 * np.random.rand(2)             
                #as well we have to change the  to some c*radii
                tmp += 1             
                  
        agents_escaped[k+1] = tmp

    return y, agents_escaped, a

def exp_euler(y0, v0, f, N_steps, dt, room):
    tmp = 0
    agents_escaped = np.zeros(N_steps)

    y = np.zeros((y0.shape[0], y0.shape[1], N_steps))
    v = np.zeros((y0.shape[0], y0.shape[1], N_steps))
    a = np.zeros((y0.shape[0], y0.shape[1], N_steps))

    y[:,:,0] = y0
    v[:,:,0] = v0

    for k in range(N_steps-1):
        #print(100*k/N_steps, '% done.')
        a[:,:,k] = f(y[:,:,k], v[:,:,k])
        v[:,:,k+1] = v[:,:,k] + dt*a[:,:,k]
        y[:,:,k+1] = y[:,:,k] + dt*v[:,:,k+1]

        for i in range(y.shape[1]):
            # checks if there are two destination and calculates the distance to the closets destination
            destination = np.zeros(len(room.get_destination()))
            for count, des in enumerate(room.get_destination()):
                destination[count] = np.linalg.norm(y[:, i, k + 1] - des)
            distance = np.amin(destination)

            if distance < 0.5:
                #we have to use position of door here instead of (0,5)
                y[:,i,k+1] = 10**6 * np.random.rand(2)             
                #as well we have to change the  to some c*radii
                tmp += 1             
            
        agents_escaped[k+1] = tmp

    return y, agents_escaped, a

def leap_frog(y0, v0, f, N_steps, dt, room):
    tmp = 0
    agents_escaped = np.zeros(N_steps-1)

    y = np.zeros((y0.shape[0], y0.shape[1], N_steps))
    v = np.zeros((y0.shape[0], y0.shape[1], N_steps))
    a = np.zeros((y0.shape[0], y0.shape[1], N_steps))
    
    y[:,:,0] = y0
    #v[:,:,0] += 0.5*dt*f(y[:,:,0], v[:,:,0])
    v[:,:,0] = v0 + 0.5*dt*f(y0)
   
    for k in range(N_steps-1):
        #print(100*k/N_steps, '% done.')
        y[:,:,k+1] = y[:,:,k] + dt*v[:,:,k]
        a[:,:,k] = f(y[:,:,k], v[:,:,k])
        v[:,:,k+1] = v[:,:,k] + dt*f(y[:,:,k+1], v[:,:,k] + dt*a[:,:,k])

        for i in range(y.shape[1]):
            # checks if there are two destination and calculates the distance to the closets destination
            destination = np.zeros(len(room.get_destination()))
            for count, des in enumerate(room.get_destination()):
                destination[count] = np.linalg.norm(y[:, i, k + 1] - des)
            distance = np.amin(destination)

            if distance < 0.5:
                #we have to use position of door here instead of (0,5)
                y[:,i,k+1] = 10**6 * np.random.rand(2)             
                #as well we have to change the  to some c*radii
                tmp += 1             
       
        agents_escaped[k+1] = tmp

    return y, agents_escaped, a

def leap_frog_v2(y0, v0, f, N_steps, dt, room, it=500):
    tmp = 0
    agents_escaped = np.zeros(N_steps)

    y = np.zeros((y0.shape[0], y0.shape[1], N_steps))
    v = np.zeros((y0.shape[0], y0.shape[1], N_steps))
    a = np.zeros((y0.shape[0], y0.shape[1], N_steps))
    
    y[:,:,0] = y0
    v[:,:,0] = v0 + 0.5*dt*f(y0, v0)
    a[:,:,0] = f(y0, v0)

    for k in range(N_steps-1):
        #print(100*k/N_steps, '% done.')
        ytemp = y[:,:,k]
        vtemp = v[:,:,k]

        for l in range(it-1):
            ytemp += dt/it*vtemp
            vtemp += dt/it*f(ytemp, vtemp + dt/it*f(ytemp, vtemp))

        y[:,:,k+1] = ytemp + dt/it*vtemp
        a[:,:,k+1] = f(ytemp, vtemp)
        v[:,:,k+1] = vtemp + dt*f(ytemp, vtemp + dt*a[:,:,k+1])

        for i in range(y.shape[1]):
            # checks if there are two destination and calculates the distance to the closets destination
            destination = np.zeros(len(room.get_destination()))
            for count, des in enumerate(room.get_destination()):
                destination[count] = np.linalg.norm(y[:, i, k + 1] - des)
            distance = np.amin(destination)

            if distance < 0.5:
                #we have to use position of door here instead of (0,5)
                y[:,i,k+1] = 10**6 * np.random.rand(2)             
                #as well we have to change the  to some c*radii
                tmp += 1             
        
        agents_escaped[k+1] = tmp
    
    return y, agents_escaped, a

"""
def mcstep(y0, v0, f, t, dt, room, samples, iterations):
    y = np.zeros((y0.shape[0], y0.shape[1], samples))
    v = np.zeros((y0.shape[0], y0.shape[1], samples))
    parameters = np.sort(np.random.uniform(low = 0.0, high = dt, size = (iterations-1, samples)), axis=0)
    p = np.zeros((iterations, samples))
    p[0,:], p[-1,:] = parameters[0,:], 1-parameters[-1,:]
    p[1:-1,:] = parameters[1:,:] - parameters[:-1,:] 

    for i in range(samples):
        ytemp = y0
        vtemp = v0

        for j in range(iterations):
            vtemp += p[j,i]*f(ytemp, vtemp)
            ytemp += p[j,i]*vtemp  
            
        y[:,:,k] = ytemp
        v[:,:,k] = vtemp    

    return np.sum(y, axis=2)/samples, np.sum(v, axis=2)/samples

def monte_carlo(y0, v0, f, N_steps, dt, room, samples=40, iterations=50):
    tmp = int(0)
    agents_escaped = np.zeros(N_steps)
    y = np.zeros((y0.shape[0], y0.shape[1], N_steps))
    v = np.zeros((y0.shape[0], y0.shape[1], N_steps))
    a = np.zeros((y0.shape[0], y0.shape[1], N_steps))

    y[:,:,0] = y0
    v[:,:,0] = v0
    a[:,:,0] = f(y0, v0)

    for k in range(N_steps-1):
        y[:,:,k+1], v[:,:,k+1] = mcstep(y[:,:,k], v[:,:,k], f, 0.0, 0.5, room, samples, iterations)
        a[:,:,k+1] = f(y[:,:,k+1], v[:,:,k+1])

        for i in range(y.shape[1]):
            # checks if there are two destination and calculates the distance to the closets destination
            destination = np.zeros(len(room.get_destination()))
            for count, des in enumerate(room.get_destination()):
                destination[count] = np.linalg.norm(y[:, i, k + 1] - des)
            distance = np.amin(destination)

            if distance < 0.5:
                #we have to use position of door here instead of (0,5)
                y[:,i,k+1] = 10**6 * np.random.rand(2)             
                #as well we have to change the  to some c*radii
                tmp += 1  
       
        agents_escaped[k+1] = tmp
   
    return y, agents_escaped, a#, flag
"""   

def odestep(y0, v0, f, t, dt, room, dtmin = 0.0001, tol = 0.00001, maxiter = 100000):
    ytemp = y0
    vtemp = v0
    it = 0
    step = t
    dx = 0.1*dt

    a21 = (1.0/5.0)
    a31 = (3.0/40.0)
    a32 = (9.0/40.0)
    a41 = (44.0/45.0)
    a42 = (-56.0/15.0)
    a43 = (32.0/9.0)
    a51 = (19372.0/6561.0)
    a52 = (-25360.0/2187.0)
    a53 = (64448.0/6561.0)
    a54 = (-212.0/729.0)
    a61 = (9017.0/3168.0)
    a62 = (-355.0/33.0)
    a63 = (46732.0/5247.0)
    a64 = (49.0/176.0)
    a65 = (-5103.0/18656.0)
    a71 = (35.0/384.0)
    a73 = (500.0/1113.0)
    a74 = (125.0/192.0)
    a75 = (-2187.0/6784.0)
    a76 = (11.0/84.0)
    a81 = (5179.0/57600.0)
    a83 = (7571.0/16695.0)
    a84 = (393.0/640.0)
    a85 = (-92097.0/339200.0)
    a86 = (187.0/2100.0)
    a87 = (1.0/40.0)

    for i in range(maxiter):
        v1 = vtemp
        k1 = f(ytemp, vtemp)
        
        v2 = a21*v1
        k2 = f(ytemp + dx*v2, vtemp + dx*a21*k1)
        
        v3 = a31*v1 + a32*v2
        k3 = f(ytemp + dx*v3, vtemp + dx*(a31*k1 + a32*k2))
        
        v4 = a41*v1 + a42*v2 + a43*v3
        k4 = f(ytemp + dx*v4, vtemp + dx*(a41*k1 + a42*k2 + a43*k3))
        
        v5 = a51*v1 + a52*v2 + a53*v3 + a54*v4
        k5 = f(ytemp + dx*v5, vtemp + dx*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
        
        v6 = a61*v1 + a62*v2 + a63*v3 + a64*v4 + a65*v5
        k6 = f(ytemp + dx*v6, vtemp + dx*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))
        
        v7 = a71*v1 + a73*v3 + a74*v4 + a75*v5 + a76*v6
        k7 = f(ytemp + dx*v7, vtemp + dx*(a71*k1 + a73*k3 + a74*k4 + a75*k5 + a76*k6)) 
        
        #Error Control
        error = np.linalg.norm((a71-a81)*k1 + (a73-a83)*k3 + (a74-a84)*k4 + (a75-a85)*k5 + (a76-a86)*k6 - a87*k7)
        delta = 0.84*((tol/error)**0.2)
        
        
        if (error < tol or dx <= dtmin):
            step += dx
            vtemp += dx*(a81*k1 + a83*k3 + a84*k4 + a85*k5 + a86*k6 + a87*k7)
            ytemp += dx*(a81*v1 + a83*v3 + a84*v4 + a85*v5 + a86*v6 + a87*v7)

        if (delta <= 0.1):
            dx *= 0.1
        elif (delta >= 4.0):
            dx *= 2.0
        else:
            dx *= delta

        if (dx >= 0.5*dt):
            dx = 0.5*dt

        if (step >= t + dt):
            break
        elif (step + dx > t + dt):
            dx = t + dt - step
            continue
        elif (dx < dtmin):
            dx = dtmin
            continue    
        
    return ytemp, vtemp

def ode45(y0, v0, f, N_steps, dt, room):
    tmp = int(0)
    agents_escaped = np.zeros(N_steps)
    y = np.zeros((y0.shape[0], y0.shape[1], N_steps))
    v = np.zeros((y0.shape[0], y0.shape[1], N_steps))
    a = np.zeros((y0.shape[0], y0.shape[1], N_steps))

    y[:,:,0] = y0
    v[:,:,0] = v0
    a[:,:,0] = f(y0, v0)
   
    for k in range(N_steps-1):
        y[:,:,k+1], v[:,:,k+1] = odestep(y[:,:,k], v[:,:,k], f, 0.0, 0.5, room)
        a[:,:,k+1] = f(y[:,:,k+1], v[:,:,k+1])

        for i in range(y.shape[1]):
            # checks if there are two destination and calculates the distance to the closets destination
            destination = np.zeros(len(room.get_destination()))
            for count, des in enumerate(room.get_destination()):
                destination[count] = np.linalg.norm(y[:, i, k + 1] - des)
            distance = np.amin(destination)

            if distance < 0.5:
                #we have to use position of door here instead of (0,5)
                y[:,i,k+1] = 10**6 * np.random.rand(2)             
                #as well we have to change the  to some c*radii
                tmp += 1             
       
        agents_escaped[k+1] = tmp
    
    return y, agents_escaped, a

# -------------------------------------------------------- DiffEq.py-----------------------------------------------------------------

class Diff_Equ:
    def __init__(self, num_individuals, L, tau, room, radii, weights):
        # Initial conditions
        self.room = room                            #import class room as room
        self.N = num_individuals                    # quantity of agents
        self.m = weights                            # mass of agents (kg)
        self.v_0 = 1.5 * np.ones(self.N)            # desired velocity (m/s)
        self.radii = radii                          # radii of agents (m)
        self.A = 2*10**3                            # constant (N)
        self.B = 0.08                               # constant (m)
        self.tau = tau                              # time-step (s)
        self.k = 1.2*10**5                          # parameter (kg/s^2)
        self.kap = 2.4*10**5                        # parameter (kg/(m*s^2))
        self.L = L                                  # size of square room (m)
        self.r_D = room.get_destination()           # position of door (m,m)
        self.numwalls = self.room.get_num_walls()   # number of walls 
        self.walls = self.room.walls                # position of corner points of the walls
        self.wallshere = self.room.wallshere        # True if there are walls in the middle of the room

    #Checks if an agent touches anotherone or a wall  
    def g(self, x):        
        """returns a positive float, the absolute value if and only if the input is negative
             parameters:   ``x``: float
        """    
        if x < 0:
            return 0
        else:
            return x      
        
    #Adds the radii of agents i and j
    def rad(self, i, j):  
        """ returns a float, the sum of two radiis
            parameters: ``i``,``j``: integers (agent i and j)
        """
        return self.radii[i] + self.radii[j]

    #Distance between agent i at positon r and wall j
    def wall_distance(self, i, j, r):
        """ returns:    ``distance``: float (smallest distance between agent i and wall j)
                        ``n``, ``t``: 1D-array (normalized vector pointing from 
                                      wall to agent, and it's tangential vector)
                        ``nearest`` : 1D-array (point on the wall closest to agent)
            paramters:  ``i``, ``j``: integers (agent i and wall j)
                        ``r``       : 2D-array (positions of all agents)
        """
        temp_wall      = self.walls[j,:,:]
        line_vec       = temp_wall[1,:]-temp_wall[0,:]
        pnt_vec        = r[:,i]-temp_wall[0,:]
        line_len       = np.linalg.norm(line_vec)
        line_unitvec   = line_vec/line_len
        pnt_vec_scaled = pnt_vec/line_len
        temp           = line_unitvec.dot(pnt_vec_scaled)    
        if temp < 0.0:
            temp = 0.0
        elif temp > 1.0:
            temp = 1.0
        nearest       = line_vec*temp
        dist          = pnt_vec-nearest
        nearest       = nearest + temp_wall[0,:]
        distance      = np.linalg.norm(dist)
        n             = dist/distance
        t             = np.array([-n[1], n[0]])
        return distance, n, t, nearest   
    
    #Distance between agent i and agent j
    def agent(self, i, j, r,v):
        """ returns:    ``d``      : float (distance between agend i and j)
                        ``n``,``t``: 1D-array (normalized vector pointing from
                                     agent i to j and it's tangential vector)
                        ``dv_t``   : float (difference in velocity)
            parameters: ``i``,``j``: integers (agent i and j)
                        ``r``,``v``: 2D-array (position and velocity of all agents)
        """
        d    = np.linalg.norm(r[:, i] - r[:, j])
        n    = (r[:, i] - r[:, j])/d
        t    = np.array([-n[1],n[0]])
        temp = v[:, j] - v[:, i]
        dv_t = temp.dot(t)
        return d, n, t, dv_t
    
    #Force between agent i and j
    def f_ij(self, i, j, r, v):  
        """ returns an 1D-array, the frocevector between agent i and j
            parameters: ``i``,``j``: integers (agent i and j)
                        ``r``,``v``: 2D-array (position and velocity of all agents)
        """
        d, n, t, dv_t = self.agent(i, j, r,v)
        rad_ij = self.rad(i,j)
        a = self.A * np.exp((rad_ij - d) / self.B) + self.k * self.g(rad_ij - d)
        b = self.kap * self.g(rad_ij - d) * dv_t
        return a * n + b * t
    
    #Force between agent i and wall j
    def f_iW(self, i, j, r, v):  
        """ returns an 1D-array, the frocevector between agent i and wall j
            parameters: ``i``,``j``: integers (agent i and wall j)
                        ``r``,``v``: 2D-array (position and velocity of all agents)
        """
        d,n,t = self.wall_distance(i, j, r)[:-1]
        a = self.A * np.exp((self.radii[i] - d) / self.B) + self.k * self.g(self.radii[i] - d)
        b = self.kap * self.g(self.radii[i] - d) * v[:, i].dot(t)
        return a * n - b * t

    #point of intersection of two lines formed by the points (a1,a2), respectively (b1,b2)
    def seg_intersect(self, a1,a2,b1,b2):
        da = a2-a1
        db = b2-b1
        dp = a1-b1
        dap = np.array([-da[0][1],da[0][0]])
        denom = np.dot(dap, db)
        num = np.dot(dap, dp)
        return (num / denom.astype(float))*db + b1

    #return true if the point c is between the two points a and b
    def is_between(self,a,b,c):
        return np.linalg.norm(a-c) + np.linalg.norm(c-b) == np.linalg.norm(a-b) 
    
    #taking the right agent vector direction
    def direction(self,i,j,r):
        wall = self.walls[j,:,:]
        wall_norm = (wall[0,:]-wall[1,:])/np.linalg.norm(wall[0,:]-wall[1,:])
        point = self.seg_intersect(r[:,i],self.r_D,wall[0,:],wall[1,:])
        t_or_f = self.is_between(wall[0,:],wall[1,:],point)
        
        if (t_or_f == 1 and np.linalg.norm(-r[:,i]+self.r_D)+self.radii[i] > np.linalg.norm(-point+self.r_D)) or (np.min([np.linalg.norm(r[:,i]-wall[0,:]),np.linalg.norm(r[:,i]-wall[1,:])]) < np.linalg.norm(2*self.radii[i]) and np.linalg.norm(-r[:,i]+self.r_D) > np.linalg.norm(-point+self.r_D)):
            #if the internal walls form corners, use this
            #e = self.e_1(r[:,i],wall,i,j)
            #if the internal walls does not form corners, that's quicker 
            e = self.nearest_path(wall,t_or_f,point,wall_norm,r[:,i],i)
        else:
            e = self.e_0(r[:,i],i)
        return e 

    #take the right direction in case of wall corner points
    def e_1(self,r_i,temp_wall,i,j):
        all_walls = self.walls[:,:,:]
        wall_norm = (temp_wall[0,:]-temp_wall[1,:])/np.linalg.norm(temp_wall[0,:]-temp_wall[1,:])
        check_close_corner = np.zeros((all_walls.shape[0],2))
        point = self.seg_intersect(r_i,self.r_D,temp_wall[0,:],temp_wall[1,:])
        t_or_f = self.is_between(temp_wall[0,:],temp_wall[1,:],point)
        
        for k in range(all_walls.shape[0]):
            if k == j:
                continue
            check_close_corner[k,0] = self.is_between(all_walls[k,0,:],all_walls[k,1,:],temp_wall[0,:])
            check_close_corner[k,1] = self.is_between(all_walls[k,0,:],all_walls[k,1,:],temp_wall[1,:])
        
        if np.sum(check_close_corner) == 1:
            for k in range(all_walls.shape[0]):
                if (check_close_corner[k,0] == 1 and check_close_corner[k,1] == 0) and self.is_between(all_walls[k,0,:],all_walls[k,1,:],self.seg_intersect(r_i,self.r_D,all_walls[k,0,:],all_walls[k,1,:])) == 1:
                    e = self.nearest_path(temp_wall,t_or_f,point,wall_norm,r_i,i)
                    break
                elif (check_close_corner[k,0] == 1 and check_close_corner[k,1] == 0) and self.is_between(all_walls[k,0,:],all_walls[k,1,:],self.seg_intersect(r_i,self.r_D,all_walls[k,0,:],all_walls[k,1,:])) == 0:
                    if np.linalg.norm(r_i-point)<self.radii[i]:
                        e = self.nearest_path(temp_wall,t_or_f,point,wall_norm,r_i,i)
                    elif (t_or_f == 1 and np.linalg.norm(point-r_i)<2*self.radii[i]):
                        e =  - wall_norm 
                        break
                    else:
                        p = temp_wall[1,:] - wall_norm*2*self.radii[i]
                        e = (-r_i+p)/np.linalg.norm(-r_i+p)
                        break
                elif check_close_corner[k,0] == 0 and check_close_corner[k,1] == 1 and self.is_between(all_walls[k,0,:],all_walls[k,1,:],self.seg_intersect(r_i,self.r_D,all_walls[k,0,:],all_walls[k,1,:])) == 1:
                    e = self.nearest_path(temp_wall,t_or_f,point,wall_norm,r_i,i)
                    break
                    
                elif check_close_corner[k,0] == 0 and check_close_corner[k,1] == 1 and self.is_between(all_walls[k,0,:],all_walls[k,1,:],self.seg_intersect(r_i,self.r_D,all_walls[k,0,:],all_walls[k,1,:])) == 0:    
                    if np.linalg.norm(r_i-point)<self.radii[i]:
                        e = self.nearest_path(temp_wall,t_or_f,point,wall_norm,r_i,i)
                    elif (t_or_f == 1 and np.linalg.norm(point-r_i)<2*self.radii[i]):
                        e =  wall_norm 
                        break
                    else: 
                        p = temp_wall[0,:] + wall_norm*2*self.radii[i]
                        e = (-r_i+p)/np.linalg.norm(-r_i+p)
                        break
        else:
            e = self.nearest_path(temp_wall,t_or_f,point,wall_norm,r_i,i)
        return e
    
    
    #take the nearest path if one agents has to overtake a wall
    def nearest_path(self,temp_wall,t_or_f,point,wall_norm,r_i,i):
        if (np.linalg.norm(self.r_D-temp_wall[0,:]) + np.linalg.norm(r_i-temp_wall[0,:])) <= (np.linalg.norm(self.r_D-temp_wall[1,:]) + np.linalg.norm(r_i-temp_wall[1,:])):
            #if (t_or_f == 1 and np.linalg.norm(point-r_i)<2*self.radii[i]):
            #    e =  wall_norm 
            #else: 
            p = temp_wall[0,:] + wall_norm*2*self.radii[i]
            e = (-r_i+p)/np.linalg.norm(-r_i+p)
        else:
            #if (t_or_f == 1 and np.linalg.norm(point-r_i)<2*self.radii[i]):
            #    e =  - wall_norm 
            #else:
            p = temp_wall[1,:] - wall_norm*2*self.radii[i]
            e = (-r_i+p)/np.linalg.norm(-r_i+p)
        return e
    

    #Desired direction normalized for one agent
    def e_0(self, r, i):  
        """ returns an 1D-array, the normalized desired direction of agent i
            parameters: ``i``: integer (agent i)
                        ``r``: 1D-array (position of agent i)
        """
        #If there are two destinations, then half of the people go to each destination
        if len(self.r_D) == 2:
            if i < self.N/2:
                return (-r + self.r_D[0]) / np.linalg.norm(-r + self.r_D[0])
            else:
                return (-r + self.r_D[1]) / np.linalg.norm(-r + self.r_D[1])

        return (-r + self.r_D) / np.linalg.norm(-r + self.r_D)

    #Finding the nearest wall that ubstruct one person
    def nearest_wall(self,r_i):
        all_walls = self.walls[5:,:,:]
        distance = np.zeros((all_walls.shape[0]))
        for i in range(all_walls.shape[0]):
            temp_wall = all_walls[i,:,:]
            point = self.seg_intersect(r_i,self.r_D,temp_wall[0,:],temp_wall[1,:])
            distance[i] = np.linalg.norm(point-r_i)
        return 5+np.argmin(distance)
  

    #Desired direction normalized for all agents
    def e_t(self, r):
        """ returns an 1D-array, the normalized desired direction at the actual 
            timestep for all agents
            parameters: ``r``: 2D-array (position of all agents)
        """
        e_temp = np.zeros((2, self.N))
        #If there are additional walls the desired direction doesn't have to...
        #...be the direction of a door.
        if self.wallshere == False: 
            for i in range(self.N):
                e_temp[:, i] = self.e_0(r[:,i], i)
        else:
            for i in range(self.N):
                j = self.nearest_wall(r[:,i])
                e_temp[:, i] = self.direction(i,j,r)
        return e_temp


    #The interacting force of the agents to each other  
    def f_ag(self, r, v):
        """ returns a 3D-array, the interacting forces between all the agents
            parameters: ``r``,``v``: 2D-array (position and velocity of all agents) 
        """
        f_agent = np.zeros((2, self.N))
        fij     = np.zeros(((2, self.N, self.N)))
        for i in range(self.N-1):
            for j in range(self.N-1-i):
                    fij[:,i,j+i+1] = self.f_ij(i,j+i+1,r,v)
                    fij[:,j+i+1,i] = -fij[:,i,j+i+1]
        f_agent = np.sum(fij, 2)
        return f_agent

    #The force of each wall acting on each agents
    def f_wa(self, r, v):
        """ returns a 3D-array, the interacting forces between all agents and walls
            parameters: ``r``,``v``: 2D-array (position and velocity of all agents) 
        """
        f_wall = np.zeros((2, self.N))
        for i in range(self.N):
            for j in range(self.numwalls):
                f_wall[:, i] += self.f_iW(i, j, r, v)
        return f_wall
    
    #The diff_equation of our problem
    #Calculates the accelaration of each agent
    def f(self, r, v):
        """ returns a 2D-array 
        v = the velocity at time t
        r = the position at time t"""
        e_temp = self.e_t(r)
        acc = (self.v_0 * e_temp - v) / self.tau + self.f_ag(r, v) / self.m + self.f_wa(r, v) / self.m
        return acc

# -------------------------------------------------------- Simulation.py ------------------------------------------------------------
class Simulation:
    def __init__(self, num_individuals, num_steps, method=leap_frog, tau=0.1, v_des=1.5, room="square",
                 room_size=25):

        std_deviation = 0.07                    
        variation = np.random.normal(loc=1, scale=std_deviation, size=(1, num_individuals)) # is late used to make the agents differ in weight and size

        # Constants
        self.L = room_size  # size of square room (m)
        self.N = num_individuals  # quantity of pedestrians
        self.tau = tau  # time-step (s)
        self.num_steps = num_steps  # number of steps for integration

        # Agent information
        self.radii = 0.4 * (np.ones(self.N)*variation).squeeze()  # radii of pedestrians (m)
        self.v_des = v_des * np.ones(self.N)  # desired velocity (m/s)
        self.m = 80 * (np.ones(self.N)*variation).squeeze()  # mass of pedestrians (kg)
        self.forces = None              # forces on the agents
        self.agents_escaped = None    #number of agents escaped by timesteps
        self.v = np.zeros((2, self.N, self.num_steps))  # Three dimensional array of velocity
        self.y = np.zeros(
            (2, self.N, self.num_steps))  # Three dimensional array of place: x = coordinates, y = Agent, z=Time
        
        # other
        self.room = Room(room, room_size)  # kind of room the simulation runs in
        self.method = method  # method used for integration
        self.diff_equ = Diff_Equ(self.N, self.L, self.tau, self.room, self.radii, self.m)  # initialize Differential equation
    
    # function set_time, set_steps give the possiblity to late change these variable when needed
    def set_steps(self, steps):
        self.num_steps = steps

    # function to change the methode of integration if needed
    def set_methode(self, method):
        self.method = method

    def dont_touch(self, i, x):  # yields false if people don't touch each other and true if they do
        for j in range(i - 1):
            if np.linalg.norm(x - self.y[:, j, 0]) < 3 * self.radii[i]:
                return True
        return False

    # fills the spawn zone with agents with random positions
    def fill_room(self):
        spawn = self.room.get_spawn_zone()
        len_right = spawn[0, 1] - spawn[0, 0]
        len_left = spawn[1, 1] - spawn[1, 0]
        max_len = max(len_left, len_right)

        # checks if the area is too small for the agents to fit in
        area_people = 0
        for i in range(self.N):
            area_people += 4 * self.radii[i] ** 2
        if area_people >= 0.7 * max_len ** 2:
            sys.exit('Too many people! Please change the size of the room/spawn-zone or the amount of people.')
        # checks if the agent touches another agent/wall and if so gives it a new random position in the spawn-zone 
        for i in range(self.N):
            # The pedestrians don't touch the wall
            x = len_right*np.random.rand() + spawn[0, 0]
            y = len_left * np.random.rand() + spawn[1, 0]
            pos = [x, y]

            # The pedestrians don't touch each other
            while self.dont_touch(i, x):
                x = len_right * np.random.rand() + spawn[0, 0]
                y = len_left * np.random.rand() + spawn[1, 0]
                pos = [x, y]
            self.y[:, i, 0] = pos

        self.v[:, :, 0] = self.v_des * self.diff_equ.e_t(self.y[:, :, 0])

    # calls the method of integration with the starting positions, diffequatial equation, number of steps, and delta t = tau
    def run(self):
        self.y, self.agents_escaped, self.forces = self.method(self.y[:, :, 0], self.v[:, :, 0], self.diff_equ.f, self.num_steps, self.tau, self.room)

    # Displayes the simulation in pygame
    def get_data(self):
        return self.y, self.agents_escaped, self.forces


# ----------------------------------------------------------------------------------------------------------------------------------

if __name__=='__main__':
    
    sim1 = Simulation(8, 200, ode45, room='square') # input methods as functions, NOT AS STRINGS !!!!!!!!!!!!!!!! 50, 1000
    sim1.fill_room()
    sim1.run()
    y1, agents1, forces1 = sim1.get_data()
    np.save('y.npy', y1)
    np.save('agents.npy', agents1)
    np.save('forces.npy', forces1)