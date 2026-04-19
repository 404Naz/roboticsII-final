import numpy as np
import matplotlib.pyplot as plt
import time
import math

from robot import BicycleRobot
from scipy.interpolate import interp1d

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
    timer = 0
    while (running and timer < 120): # limit each sim to 2 mins
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

        timer += delta_t

    for robot, ax in zip(robots_lst, axes):
        x,y = zip(*robot.detected_obs)
        ax.scatter(x, y, s=10, c=robot.color, marker='x')
        print(f"{robot.color} robot time: {robot.timer}ms with path len: {robot.path_len}")

    sim_res = dict()
    sim_distance_to_objects = dict()
    sim_err_over_time = dict()
    sim_time = dict()
    for robot in robots_lst:
        sim_distance_to_objects[robot.name] = robot.distance_to_closest_object
        sim_err_over_time[robot.name] = robot.error_over_time
        sim_time[robot.name] = robot.timer
    sim_res["DistToObj"] = sim_distance_to_objects
    sim_res["ErrorOverTime"] = sim_err_over_time
    sim_res["time"] = sim_time
    return sim_res

def main(num_iterations=1, initial_seed=0, scale=1):
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

    simulation_results = dict()

    plt.ion()

    for i in range(num_iterations):
        obstacles = []
        # constistent seed for testing
        np.random.seed(initial_seed+scale*i)
        for _ in range(NUM_OBSTACLES):
            obstacles.append((np.random.uniform(XMIN,XMAX), np.random.uniform(YMIN,YMAX)))
        simulation_results[i] = simulation(robots, obstacles, i)

    # for robot in robots:
    #     fig, ax = plt.subplots()
    #     num_samples = int(robot.timer / delta_t)
    #     ax.set_xlabel("Time (s)")
    #     ax.set_ylabel("Distance (m)")
    #     ax.set_title(f"Distance (error) between true and measured position {robot.name}")
    #     ax.grid(True)
    #     for i in range(num_iterations):
    #         times = np.linspace(0, simulation_results[i]["time"][robot.name], num_samples)
    #         ax.plot(times, simulation_results[i]["ErrorOverTime"][robot.name], label=f"Iteration {i}")

    COMMON_N = 200
    t_common = np.linspace(0,1,COMMON_N)

    for robot in robots:
        fig, ax = plt.subplots()
        ax.set_xlabel("Normalized Time")
        ax.set_ylabel("Distance (m)")
        ax.set_title(f"Distance to closest obstacle for {robot.name}")
        ax.grid(True)
        all_resampled = []
        for i in range(num_iterations):
            # plotting help by copilot 04/18/2026
            dist = simulation_results[i]["DistToObj"][robot.name]
            N = len(dist)
            t_norm = np.linspace(0,1,N)
            f = interp1d(t_norm, dist, kind='linear')
            dist_resampled = f(t_common)
            all_resampled.append(dist_resampled)
            ax.plot(t_common, dist_resampled, label=f"Iteration {i}")

        mean_curve = np.mean(np.vstack(all_resampled), axis=0)
        ax.plot(t_common, mean_curve, color='black', linewidth=2, label="Mean")

        plt.legend()

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    NUM_ITERATIONS = 2
    INIT_SEED = 0
    SCALE = 1
    main(num_iterations=NUM_ITERATIONS, initial_seed=INIT_SEED, scale=SCALE)
