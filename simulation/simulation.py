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

timestep = 0
timesteps = []
true_pos = []
meas_pos = []

goal_position = np.array([-30.0, 40.0])

R = np.array([[0.005, 0, 0], [0, 0.005, 0], [0, 0, 0]])
Q = np.array([[0.1, 0, 0], [0, 0.1,0], [0, 0, 0.001]])

copied = False

particles = []
obstacles = []

robots = [
    BicycleRobot(name="B1", color="green", w=WIDTH, h=HEIGHT, L=L, x=0, y=0, r=0, loc_particles=particles, R=R, Q=Q, detect_range=5, detect_fov_deg=40),
    BicycleRobot(name="B2", color="pink", w=WIDTH, h=HEIGHT, L=L, x=0, y=0, r=0, loc_particles=particles, R=R, Q=Q, detect_range=5, detect_fov_deg=80),
    # BicycleRobot(name="B3", color="orange",w=WIDTH, h=HEIGHT, L=L, x=0, y=0, r=0, loc_particles=particles, R=R, Q=Q, detect_range=10, detect_fov_deg=40),
    # BicycleRobot(name="B4", color="blue",w=WIDTH, h=HEIGHT, L=L, x=0, y=0, r=0, loc_particles=particles, R=R, Q=Q, detect_range=10, detect_fov_deg=80),
]

print("Generating particles")

# constistent seed for testing
# np.random.seed(0)
for i in range(NUM_PARTICLES):
    particles.append((np.random.uniform(-10,10), np.random.uniform(-10,10), np.random.uniform(-np.pi/12,np.pi/12)))

for i in range(NUM_OBSTACLES):
    obstacles.append((np.random.uniform(XMIN,XMAX), np.random.uniform(YMIN,YMAX)))

x,y = zip(*obstacles)

print("Done generating particles")


num_robots = len(robots)
plt.ion()
fig, axes = plt.subplots(1, num_robots, figsize=(6*num_robots, 6))

robot_plots = []
for robot, ax in zip(robots, axes):
    ax.set_xlim([XMIN, XMAX])
    ax.set_ylim([YMIN, YMAX])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True)
    ax.scatter(goal_position[0], goal_position[1], s=100, c='k', marker='x')
    ax.scatter(x, y, s=10, color='red', marker='o')
    robot_plots.append(robot.draw_robot(ax))

running = True

while (running):
    plt.pause(PAUSE_TIME)
    time.sleep(PAUSE_TIME)

    running = False
    
    for i, robot in enumerate(robots):
        for artist in robot_plots[i]:
            artist.remove()
        # Plot trajectory point
        axes[i].scatter(robot.true_pos[0], robot.true_pos[1], s=2, c=robot.color)
        # control robot
        running |= robot.controller(goal_position=goal_position, obstacles=obstacles, dt=delta_t)
        # redraw robot
        robot_plots[i] = robot.draw_robot(axes[i])


    timestep += delta_t

for robot, ax in zip(robots, axes):
    x,y = zip(*robot.detected_obs)
    ax.scatter(x, y, s=10, c=robot.color, marker='x')
    print(f"{robot.color} robot time: {robot.timer}ms with path len: {robot.path_len}")

plt.ioff()
plt.show()
