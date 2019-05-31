# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 17:06:13 2018

@author: MATHIAS 
"""

import numpy as np
import matplotlib.pyplot as plt
import sys


#Initial conditions
N   = 30                        #quantity of pedestrians
m   = 80*np.ones(N)             #mass of pedestrians (kg)
radii = 0.4* np.ones(N)         #radii of pedestrians (m)
v_0 = 1.5*np.ones(N)            #desired velocity (m/s)
A   = 2* 10**3                  #constant (N)
B   = 0.08                      #constant (m)
tau = 0.5                       #time-step (s) 
k   = 1.2* 10**5                #parameter (kg/s^2) 
kap = 2.4* 10**5                #parameter (kg/(m*s^2))
L   = 10                        #size of square room (m)
r_D = np.array([0,L/2])         #position of door (m,m) 
r_W = np.array([[0,0], [0,L], [L,L], [L,0]]) # Corner points of the Wall
#numeration of walls: 0 for (x,y_min), 1 for (x_max, y), 2 for (x, y_max) and 4 for (x_min, y)


"""If the room is too small for the chosen amount of people:"""
area_people = 0 
for i in range(N):
    area_people += 4* radii[i]**2
if area_people >= 0.7*L**2:
    sys.exit('Too much people! Please change the size of the room or the amount of people.')
    
""""Initial positions"""
r_0 = np.zeros((2,N))

def dont_touch(i, r): #yields true if people don't touch each other and false if they do
    ''' 
    made it a little shorter
    
    temp = True
    for j in range(i-1):
        if (np.linalg.norm(x - r[:,j]) < 2*radii[i]):
            temp = False
        else:
            continue
    return temp 
    '''
    for j in range(i-1):
        if (np.linalg.norm(x - r[:,j]) < 2*radii[i]):
            return False
    return True

#Faster method but won't work if radii differ too much or N is too big.
#if we choose to do it this way we could also get rid of line 29-34

#Chooses N random numbers in the range (step_size, temp) with step_
#size two times the maximum radius
'''

r_max = np.amax(radii)
if 2*r_max*N > L:
    print("This method won't work here")

temp = L - step_size
x = np.random.choice(np.arange(r_max, temp, 2*r_max), N, replace=False)
y = np.random.choice(np.arange(r_max, temp, 2*r_max), N, replace=False)
r_0 =np.concatenate(([x], [y]), axis=0)
'''

for i in range(N):
    #The pedestrians don't touch the wall
    x = (L-2*radii[i])*np.array([np.random.rand(), np.random.rand()])
    x += radii[i]   
    #The pedestrians don't touch each other
    while(dont_touch(i,r_0) == False):
        x = (L-2*radii[i])*np.array([np.random.rand(), np.random.rand()])
        x += radii[i]
    r_0[:,i] = x
#Not the fastest way to implement. The while loop could have many iterations
#Maybe there's a better way
    
    
"""Functions              
   """         
def g(x):               #Do the pedestrians touch each other
    #return(abs(max(-x, 0)))        a little shorter, but maybe less clear
    if x < 0:           #returns the argument only if it's positive
        return 0
    else:
        return x

def rad(i,j):          #adds the radii of pedestrians i and j
    return radii[i]+radii[j]


def d(i,j,r):             #distance between pedestrians i and j
    return np.linalg.norm(r[:,i]-r[:,j])

def d_W(i,j,r):           #distance between agent i and wall j
    #only the y coordinate is important for walls on the "bottom" or "top"
    if (j == 0 or j == 2):
        return np.abs(r[1,i] - r_W[j,1])
    #only the x coordinate is important for the "left" and "right" wall
    else:
        return np.abs(r[0,i] - r_W[j,0])

def n(i, j, r):         #normalized vector pointing from agent i to agent j
    temp = (r[:,i]-r[:,j])
    return temp/np.linalg.norm(temp)

def n_W(i, j, r):       #normalized vector pointing from agent i to wall j
    if (j == 0 or j == 2):
        temp = np.array([ 0, (r[1,i] - r_W[j,1]) ])
    else:
        temp = np.array([ (r[0,i] - r_W[j,0]), 0 ])
    return temp/np.linalg.norm(temp)

def t(i, j, r):         #normalized tangetial direction between agent i and j
    return np.array([ -n(i,j,r)[1], n(i,j,r)[0] ])

def dv_t(i,j,r,v):      #normalized tangential velocity difference of pedestrians i and j
    temp = v[:,j] - v[:,i]
    return temp.dot(t(i,j,r))

def t_W(i, j, r):       #normalized tangential directin between agent i and wall j
    return np.array([ -n_W(i,j,r)[1], n_W(i,j,r)[0]])
 

                     

def f_ij(i, j, r, v): #force between pedestrians i and j
    a = A*np.exp( (rad(i,j) - d(i,j,r))/B ) + k*g(rad(i,j)- d(i,j,r))
    b = kap*g(rad(i,j) - d(i,j,r))*dv_t(i,j,r,v)
    return a*n(i,j,r) + b*t(i,j,r)

def f_iW(i, j, r, v): #force between pedestrian i and Wall j
    a = A*np.exp( (radii[i] - d_W(i,j,r))/B ) + k*g(radii[i]- d_W(i,j,r))
    b = kap * g(radii[i] - d_W(i,j,r)) * v[:,i].dot(t_W(i,j,r))
    return a*n_W(i,j,r) - b*t_W(i,j,r)


def e_0(r):             #desired direction normalized for one agent
    return (r - r_D)/np.linalg.norm(r - r_D)

def e_t(r):             #desired direction normalized for all agents         
    e_temp = np.zeros((2,N))
    for i in range(N): 
        e_temp[:,i] = e_0(r[:,i])
    return e_temp




"""Integration function"""

def f_ag(r,v):
    """The interacting force of the agents in each other, with v the velocity at time t """
    f_agent = np.zeros((2,N))
    for i in range(N):
        for j in range (N):
            if j != i:
                f_agent[:,i] += f_ij(i,j,r,v)
    return f_agent
            
def f_wa(r,v):
    """The force of the walls at time t, with v the velocity at time t"""
    f_wall = np.zeros((2,N))
    for i in range(N):
        for j in range (4):
            f_wall[:,i] += f_iW(i,j,r,v)
    return f_wall
    
    
def f(r, v):
    """ The function, which should be integrated. 
    v = the velocity at time t 
    r = the position at time t"""
    e_temp = e_t(r)
    acc = (v_0*e_temp - v)/tau + f_ag(r,v)/m + f_wa(r,v)/m
    return acc

#Constants and Definitions for integration:
N_steps = 10                   #Number of Integration Steps
y = np.zeros(((2,N,N_steps)))     #Three dimensional array of place: x = coordinates, y = Agent, z=Time
v = np.zeros(((2,N,N_steps)))     #Three dimensional array of velocity

y[:,:,0] = r_0
v[:,:,0] = v_0*e_t(r_0)




