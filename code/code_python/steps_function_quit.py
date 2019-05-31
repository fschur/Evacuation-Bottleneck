import numpy as np
import pygame
import matplotlib.pyplot as plt

# example positions for test purposes
movement_data = np.random.randint(20, 700, (2, 5, 5))
objects_pos = np.random.randint(100, 700, (2, 10))


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
