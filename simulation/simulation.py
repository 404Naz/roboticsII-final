import numpy as np
import matplotlib.pyplot as plt
import time
import math

from robot import BicycleRobot

# Initial configuration
init_pos = np.array([0.0, 0.0])
init_angle = 0.0
robot_conf = np.array([init_pos[0], init_pos[1], init_angle])

delta_t = 0.1

WIDTH = 1
HEIGHT = 1.5
L = 1.2

PAUSE_TIME = 0.001*delta_t
NUM_PARTICLES = 1000
NUM_OBSTACLES = 50

XMIN = -50
XMAX = 50
YMIN = 0
YMAX = 50

plt.ion()
fig, ax = plt.subplots()
ax.set_xlim([XMIN, XMAX])
ax.set_ylim([YMIN, YMAX])
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.grid(True)

timestep = 0
timesteps = []
true_pos = []
meas_pos = []

goal_position = np.array([-30.0, 40.0])
ax.scatter(goal_position[0], goal_position[1], s=100, c='k', marker='x')

R = np.array([[0.005, 0, 0], [0, 0.005, 0], [0, 0, 0]])
Q = np.array([[0.1, 0, 0], [0, 0.1,0], [0, 0, 0.001]])

copied = False

particles = []
obstacles = []

robot1 = BicycleRobot(w=WIDTH, h=HEIGHT, L=L, x=0, y=0, r=0, loc_particles=particles, R=R, Q=Q, detect_range=5, detect_fov_deg=40)
# robot2 = BicycleRobot(w=WIDTH, h=HEIGHT, L=L, x=0, y=0, r=0, loc_particles=particles, R=R, Q=Q, detect_range=10, detect_fov_deg=40)
robot3 = BicycleRobot(w=WIDTH, h=HEIGHT, L=L, x=0, y=0, r=0, loc_particles=particles, R=R, Q=Q, detect_range=5, detect_fov_deg=80)
# robot4 = BicycleRobot(w=WIDTH, h=HEIGHT, L=L, x=0, y=0, r=0, loc_particles=particles, R=R, Q=Q, detect_range=10, detect_fov_deg=80)
robot_plot1 = robot1.draw_robot(ax)
# robot_plot2 = robot2.draw_robot(ax)
robot_plot3 = robot3.draw_robot(ax)
# robot_plot4 = robot4.draw_robot(ax)

print("Generating particles")

# constistent seed for testing
np.random.seed(0)
for i in range(NUM_PARTICLES):
    particles.append((np.random.uniform(-10,10), np.random.uniform(-10,10), np.random.uniform(-np.pi/12,np.pi/12)))

for i in range(NUM_OBSTACLES):
    obstacles.append((np.random.uniform(XMIN,XMAX), np.random.uniform(YMIN,YMAX)))

x,y = zip(*obstacles)
ax.scatter(x, y, s=10, c='orange', marker='o')

print("Done generating particles")

while (abs(np.linalg.norm(goal_position - robot1.true_pos[:2])) > 0.5 ):
    # Plot trajectory point
    ax.scatter(robot1.true_pos[0], robot1.true_pos[1], s=2, c='green')
    # ax.scatter(robot2.true_pos[0], robot2.true_pos[1], s=2, c='blue')
    ax.scatter(robot3.true_pos[0], robot3.true_pos[1], s=2, c='pink')
    # ax.scatter(robot4.true_pos[0], robot4.true_pos[1], s=2, c='orange')
    # print(robot_conf[:2], abs(np.linalg.norm(goal_position - robot_conf[:2])))

    plt.pause(PAUSE_TIME)
    time.sleep(PAUSE_TIME)

    # control robots
    robot1.controller(goal_position=goal_position, obstacles=obstacles, dt=delta_t)
    # robot2.controller(goal_position=goal_position, obstacles=obstacles, dt=delta_t)
    robot3.controller(goal_position=goal_position, obstacles=obstacles, dt=delta_t)
    # robot4.controller(goal_position=goal_position, obstacles=obstacles, dt=delta_t)
    
    # Remove old robot drawing
    for artist in robot_plot1:
        artist.remove()
    # for artist in robot_plot2:
    #     artist.remove()
    for artist in robot_plot3:
        artist.remove()
    # for artist in robot_plot4:
    #     artist.remove()

    # Draw updated robot
    robot_plot1 = robot1.draw_robot(ax)
    # robot_plot2 = robot2.draw_robot(ax)
    robot_plot3 = robot3.draw_robot(ax)
    # robot_plot4 = robot4.draw_robot(ax)

    timestep += delta_t


x,y = zip(*robot1.detected_obs)
ax.scatter(x, y, s=10, c='green', marker='x')

x,y = zip(*robot3.detected_obs)
ax.scatter(x, y, s=10, c='pink', marker='x')

plt.ioff()
plt.show()
