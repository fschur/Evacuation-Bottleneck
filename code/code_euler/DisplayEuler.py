import numpy as np
import sys
import pygame
import matplotlib.pyplot as plt

# example positions for test purposes
# movement_data = np.random.randint(20, 700, (2, 5, 5))
# objects_pos = np.random.randint(100, 700, (2, 10))


# -------------------------------------------------------- Room.py ----------------------------------------------------------

class Room:
    def __init__(self, room, room_size):
        self.room_size = room_size
        if room == "square":
            self.door_size = room_size/15                    # size of the door is proportional to the size of the room
            self.destination = np.array([[-1, room_size/2]])   # destination the agenst want to go
            self.num_walls = 5
            self.walls = np.array([[[0, 0], [0, room_size/2-self.door_size/2]], # wall 1
                          [[0, room_size/2+self.door_size/2], [0, room_size]],  # wall 2
                          [[0, room_size], [room_size, room_size]],             # wall 3
                          [[room_size, room_size], [room_size, 0]],             # wall 4
                          [[room_size, 0], [0, 0]]])                            # wall 5
            # agents spawn with x and y position between 1 and (room_size-1) 
            self.spawn_zone = np.array([[room_size/2, room_size-1], [1, room_size-1]])

        if room == "long_room":
            self.door_size = room_size/15                    # size of the door is proportional to the size of the room
            self.room_len = room_size
            self.room_with = room_size/5
            self.destination = np.array([[-1, self.room_with/2]])   # destination the agenst want to go
            self.num_walls = 5
            self.walls = np.array([[[0, 0], [0, self.room_with/2-self.door_size/2]],        # wall 1
                          [[0, self.room_with/2+self.door_size/2], [0, self.room_with]],    # wall 2
                          [[0, self.room_with], [self.room_len, self.room_with]],           # wall 3
                          [[self.room_len, self.room_with], [self.room_len, 0]],            # wall 4
                          [[self.room_len, 0], [0, 0]]])                                    # wall 5
            # agents spawn with x and y position between 1 and (room_size-1) 
            self.spawn_zone = np.array([[1, self.room_len-1], [1, self.room_with-1]])

        if room == "long_room_v2":
            self.door_size = room_size/15                    # size of the door is proportional to the size of the room
            self.room_len = room_size
            self.room_with = room_size/5
            self.destination = np.array([[-1, self.room_with/2], [self.room_len + 1, self.room_with/2]])   # destination the agenst want to go
            self.num_walls = 6
            self.walls = np.array([[[0, 0], [0, self.room_with/2-self.door_size/2]], # wall 1
                          [[0, self.room_with/2+self.door_size/2], [0, self.room_with]],  # wall 2
                          [[0, self.room_with], [self.room_len, self.room_with]],             # wall 3
                          [[self.room_len, self.room_with], [self.room_len, self.room_with/2+self.door_size/2]],
                          [[self.room_len, self.room_with/2-self.door_size/2], [self.room_len, 0]],                 # wall 4
                          [[self.room_len, 0], [0, 0]]])                            # wall 5
            # agents spawn with x and y position between 1 and (room_size-1)
            self.spawn_zone = np.array([[1, self.room_len-1], [1, self.room_with-1]])

        if room == "edu_11":
            self.door_size = room_size/15                    # size of the door is proportional to the size of the room
            self.destination = np.array([[-1, room_size/2]])   # destination the agenst want to go
            self.num_walls = 11
            self.walls = np.array([[[0, 0], [0, room_size/2-self.door_size/2]], # wall 1
                          [[0, room_size/2+self.door_size/2], [0, room_size]],  # wall 2
                          [[0, room_size], [room_size, room_size]],             # wall 3
                          [[room_size, room_size], [room_size, 0]],             # wall 4
                          [[room_size, 0], [0, 0]],                             # wall 5
                          [[5, room_size*0.2], [5, room_size*0.4]],             # wall 6
                          [[5, room_size*0.2], [20, room_size*0.2]],            #...
                          [[20, room_size*0.2], [20, room_size*0.4]],
                          [[5, room_size*0.8], [20, room_size*0.8]],
                          [[5, room_size*0.8], [5, room_size*0.6]],
                          [[20, room_size*0.6], [20, room_size*0.8]],
                          [[20, room_size*0.6], [20, room_size*0.8]]])                          
            # agents spawn with x and y position between 1 and (room_size-1) 
            self.spawn_zone = np.array([[0.5*room_size, room_size-1], [1, room_size-1]])

        if room == "edu_1":
            self.door_size = room_size/15                    # size of the door is proportional to the size of the room
            self.destination = np.array([[-1, room_size/2]])   # destination the against want to go
            self.num_walls = 6
            self.walls = np.array([[[0, 0], [0, room_size/2-self.door_size/2]], # wall 1
                          [[0, room_size/2+self.door_size/2], [0, room_size]],  # wall 2
                          [[0, room_size], [room_size, room_size]],             # wall 3
                          [[room_size, room_size], [room_size, 0]],             # wall 4
                          [[room_size, 0], [0, 0]],                             # wall 5
                          [[6, room_size*0.3], [6, room_size*0.7]]])           # wall 6      
            # agents spawn with x and y position between 1 and (room_size-1) 
            self.spawn_zone = np.array([[room_size/2, room_size-1], [1, room_size-1]])

        if room == "edu_room":
            self.door_size = room_size/15                    # size of the door is proportional to the size of the room
            self.destination = np.array([[-1, room_size/2]])   # destination the agenst want to go
            self.num_walls = 7
            self.walls = np.array([[[0, 0], [0, room_size/2-self.door_size/2]], # wall 1
                          [[0, room_size/2+self.door_size/2], [0, room_size]],  # wall 2
                          [[0, room_size], [room_size, room_size]],             # wall 3
                          [[room_size, room_size], [room_size, 0]],             # wall 4
                          [[room_size, 0], [0, 0]],                             # wall 5
                          [[10, room_size*0.2], [5, room_size*0.4]],            # wall 6
                          [[5, room_size*0.6], [10, room_size*0.8]]])           # wall 7                 
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

# ---------------------------------------------------- steps_function_quit.py -----------------------------------------------------


def display_events(movement_data, room, wait_time, radii, sim_size, agents_escaped):
    """This function takes as input:
    movement_data: matrix with shape (x, y, z). x=2 are the number coordinates, 
    y is the number of individuals and z is the number of timesteps.
    room: instanz of the class room. We need it to draw the walls.
    radii: The radii of the individuals.
    sim_size: The size of the image on the screen later
    The function draws a map for each timestep of the room with each individuals in it.
    The code is built in the library 'pygame'"""

    # colors
    background_color = (170, 170, 170)            #grey
    people_color = (250, 0, 0)                    #red
    destination_color = (0, 128, 0)               # green
    object_color = (0, 0, 0)                      #black

    # variable for initializing pygame
    normalizer = int(sim_size/room.get_room_size()) # the ratio (size of image) / (size of actual room) 
    map_size = (room.get_room_size()*normalizer + 100,  #size of the map
                room.get_room_size()*normalizer + 100)  #plus a little free space
    wait_time = wait_time                              #time that the simultation waits between each timestep
    wait_time_after_sim = 3000                        #waittime after simulation
    movement_data_dim = movement_data.shape         
    num_persons = movement_data_dim[1]          #number of indiciduals in the simulation
    num_time_iterations = movement_data_dim[2]  #number of timesteps
    num_walls = room.get_num_walls()          #number of walls

    pygame.init()                                 #initialize the intanz
    simulate=False                                  #variable to indicate if the simulation is running
    font = pygame.font.Font(None, 32)             #create a new object of type Font(filename, size)
    worldmap = pygame.display.set_mode(map_size)
    
    while True:
        # start simulation if any key is pressed and quits pygame if told so
        for event in pygame.event.get(): 
            if event.type == pygame.KEYDOWN:
                simulate=True
            elif event.type == pygame.QUIT:
                pygame.quit()
        worldmap.fill(0)
        #This creates a new surface with text already drawn onto it
        text = font.render('Press any key to start the simulation', True, (255, 255, 255))
        #printing the text starting with a 'distance' of (100,100) from top left
        worldmap.blit(text, (100, 100))
        pygame.display.update()
        
        if simulate == True:
            # print the map for each timestep
            for t in range(num_time_iterations):
                # quit the simulation if told so
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                
                #initialize the map with background color
                worldmap.fill(background_color)
        
                #draw each peron for timestep t
                for person in range(num_persons):
                    pygame.draw.circle(worldmap, people_color, 
                                    ((normalizer*movement_data[0, person, t] + 50).astype(int),
                                    (normalizer*movement_data[1, person, t] + 50).astype(int)),
                                    int(normalizer * radii[person]), 0)
        
                #draw each object for timestep t
                for wall in range(num_walls):
                    pygame.draw.lines(worldmap, object_color, True, 
                                    normalizer*room.get_wall(wall) + 50, 2)
                # draw the destination of the agents in green
                for des in room.get_destination():
                    pygame.draw.circle(worldmap, destination_color,
                        ((normalizer * des[0] + 50).astype(int),
                        (normalizer * des[1] + 50).astype(int)),
                        7, 0)
                # prints additional information on the screen
                strf = "Number of People Escaped: " + str(int(agents_escaped[t]))
                text = font.render(strf, True, (255, 255, 255))
                # printing the text starting with a 'distance' of (10,10) from top left
                worldmap.blit(text, (10, 10))

                strd = "Number of People: " + str(num_persons)
                textd = font.render(strd, True, (255, 255, 255))
                # printing the text starting with a 'distance' of (400,10) from top left
                worldmap.blit(textd, (400, 10))

                #update the map
                pygame.display.update()
                #wait for a while before drawing new positions
                pygame.time.wait(wait_time)
            simulate=False
            text = font.render('SIMULATION FINISHED', True, (255, 255, 255))
            worldmap.blit(text, (100, 100))
            pygame.display.update()
            pygame.time.wait(wait_time_after_sim)
            
            #Uncomment to exit function instead of going back to the loop
            #pygame.quit()
            #return

#run a test with random inputs
#display_events(movement_data, objects_pos)


def display_graph(agents_escaped, acceleration, mass):
    '''Drawn 3 graphs related to the simullation. The first one is
    the number of people who escaped the room at each timestep.
    The secound one is the number of people which experience a force higher than "tol" 
    and therfore died.
    The third one shows the forces one random agents experienceses.
    ''' 
    
    tol = 700       # force at which the people die (in Newton)

    _, num_persons, num_steps = np.shape(acceleration)
    forces_new = np.zeros((num_persons, num_steps - 1))
    num_dead = np.zeros(num_steps - 1)

    # get force by multiplying the acceleration with the mass
    for person in range(num_persons):
        for t in range(num_steps - 1):
            forces_new[person, t] = np.linalg.norm(acceleration[:, person, t]) * mass[person]
            if forces_new[person, t] > tol:
                num_dead[t] += 1

    f = plt.figure(figsize=(10, 10))
    f.subplots_adjust(hspace=0.3)

    f1 = f.add_subplot(3, 1, 1)
    f1.plot(range(len(agents_escaped)), agents_escaped, 'g')
    f1.set_ylabel("num escaped")
    f1.set_xlabel("timestep")
    f1.set_title("Escape Scenario")

    f2 = f.add_subplot(3, 1, 2)
    f2.plot(range(num_steps - 1), num_dead, 'r')
    f2.set_ylabel("num dead")
    f2.set_xlabel("timestep")

    f3 = f.add_subplot(3, 1, 3)
    chosen_agent = int(num_persons/2)
    f3.plot(range(num_steps - 1), forces_new[chosen_agent, :], 'b')
    f3.set_ylabel("forces on agent")
    f3.set_xlabel("timestep")

    plt.show(f)

if __name__=='__main__':
    y = np.load('y.npy')
    forces = np.load('forces.npy')
    agents_escaped = np.load('agents.npy')

    display_graph(agents_escaped, forces, 80 * (np.ones(y.shape[1])).squeeze())
    display_events(y, Room('square', 25), 50, 0.4 * (np.ones(y.shape[1])).squeeze(), 800, agents_escaped)
