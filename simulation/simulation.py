import numpy as np
import matplotlib.pyplot as plt
import time
import math

from robot import BicycleRobot

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

goal_position = np.array([-30.0, 40.0])

R = np.array([[0.005, 0, 0], [0, 0.005, 0], [0, 0, 0]])
Q = np.array([[0.1, 0, 0], [0, 0.1,0], [0, 0, 0.001]])

def simulation(robots_lst, obstacles_lst, iteration):
    num_robots = len(robots_lst)
    plt.ion()
    figs = []
    axes = []

    robot_plots = []
    x,y = zip(*obstacles_lst)
    for i, robot in enumerate(robots_lst):
        fig, ax = plt.subplots()
        ax.set_xlim([XMIN, XMAX])
        ax.set_ylim([YMIN, YMAX])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True)
        ax.scatter(goal_position[0], goal_position[1], s=100, c='k', marker='x')
        ax.scatter(x, y, s=10, color='red', marker='o')
        robot_plots.append(robot.draw_robot(ax))
        figs.append(fig)
        axes.append(ax)

    for robot in robots_lst:
        robot.reset()

    running = True
    while (running):
        plt.pause(PAUSE_TIME)
        time.sleep(PAUSE_TIME)

        running = False
        
        for i, robot in enumerate(robots_lst):
            for artist in robot_plots[i]:
                artist.remove()
            # Plot trajectory point
            axes[i].scatter(robot.true_pos[0], robot.true_pos[1], s=2, c=robot.color)
            # control robot
            running |= robot.controller(goal_position=goal_position, obstacles=obstacles_lst, dt=delta_t)
            # redraw robot
            robot_plots[i] = robot.draw_robot(axes[i])

    for robot, ax in zip(robots_lst, axes):
        x,y = zip(*robot.detected_obs)
        ax.scatter(x, y, s=10, c=robot.color, marker='x')
        print(f"{robot.color} robot time: {robot.timer}ms with path len: {robot.path_len}")

    for robot in robots_lst:
        fig, ax = plt.subplots()
        num_samples = int(robot.timer / delta_t)
        times = np.linspace(0, robot.timer, num_samples)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance (m)")
        ax.set_title(f"Distance (error) between true and measured position {robot.name} (iteration {iteration})")
        ax.grid(True)
        ax.plot(times, robot.error_over_time)

    for robot in robots_lst:
        fig, ax = plt.subplots()
        num_samples = int(robot.timer / delta_t)
        times = np.linspace(0, robot.timer, num_samples)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance (m)")
        ax.set_title(f"Distance to closest obstacle {robot.name} (iteration {iteration})")
        ax.grid(True)
        ax.plot(times, robot.distance_to_closest_object)

    plt.ioff()
    plt.show()

def main(num_iterations, initial_seed=0):
    particles = []
    obstacles = []

    print("Generating initial particles")

    np.random.seed(initial_seed)
    for _ in range(NUM_PARTICLES):
        particles.append((np.random.uniform(-10,10), np.random.uniform(-10,10), np.random.uniform(-np.pi/12,np.pi/12)))

    robots = [
        BicycleRobot(name="B1", color="green", w=WIDTH, h=HEIGHT, L=L, x=0, y=0, r=0, loc_particles=particles.copy(), R=R, Q=Q, detect_range=5, detect_fov_deg=40),
        BicycleRobot(name="B2", color="pink", w=WIDTH, h=HEIGHT, L=L, x=0, y=0, r=0, loc_particles=particles.copy(), R=R, Q=Q, detect_range=5, detect_fov_deg=80),
        # BicycleRobot(name="B3", color="orange",w=WIDTH, h=HEIGHT, L=L, x=0, y=0, r=0, loc_particles=particles, R=R, Q=Q, detect_range=10, detect_fov_deg=40),
        # BicycleRobot(name="B4", color="blue",w=WIDTH, h=HEIGHT, L=L, x=0, y=0, r=0, loc_particles=particles, R=R, Q=Q, detect_range=10, detect_fov_deg=80),
    ]

    print("Done generating particles")

    for i in range(num_iterations):
        obstacles = []
        # constistent seed for testing
        np.random.seed(initial_seed+2*i)
        for _ in range(NUM_OBSTACLES):
            obstacles.append((np.random.uniform(XMIN,XMAX), np.random.uniform(YMIN,YMAX)))
        simulation(robots, obstacles, i)

if __name__ == '__main__':
    NUM_ITERATIONS = 5
    main(NUM_ITERATIONS)
