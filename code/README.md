# Code Folder 

The folder code_python contains code that you can run at home.
Code_euler is code for the ETH-Euler computer. For instructions see the full reproducibility test.

There are six files in code_python:

running_code.py: That is the code that you actually run in the terminal. It initializes an instance of the Simulation class

Simulation_class.py: A class which coordinates the simulation and saves the results. It contains an instance of Integrators, Room and diff_equation.

Integrators.py: A class used for the integration of the simulation.

Room.py: A class with all the rooms, for which we implemented simulation-tests.

diff_equation.py: A class for the differential equation of the forces on the agents. It is used in the Integration class.

steps_function_quit: This file is responsible for displaying the simulation and graphs.
