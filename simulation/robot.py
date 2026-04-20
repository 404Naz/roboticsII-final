import numpy as np

K_v = 0.5
K_d = 0.1
K_h = 1.0

MAX_SPEED = 2.0
MAX_TURN = np.pi/6
FIELD = 0.5
EPSILON = 1E-6

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def particle_mean(particles):
    x = 0
    y = 0
    count = 0

    for particle in particles:
        x += particle[0]
        y += particle[1]
        count += 1

    return (x / count,y / count)

def KalmanFilter(mean, cov, u_t, z_t, A_t, B_t, C_t, R_t, Q_t):
    mean_bar = A_t @ mean + B_t @ u_t
    cov_bar = A_t @ cov @ np.transpose(A_t) + R_t
    K_t = cov_bar @ np.transpose(C_t) @ np.linalg.inv(C_t@cov_bar@np.transpose(C_t) + Q_t)
    mean_t = mean_bar + K_t @ (z_t - (C_t @ mean_bar))
    cov_t = (np.eye(2) - (K_t @ C_t)) @ cov_bar
    return mean_t, cov_t

def ParticleFilter(particles, u_t, z_t, R, Q) -> list[tuple]:
    X_ = []

    # predict = [particle+u_t for particle in particles]
    predict = [particle+u_t+np.random.multivariate_normal(np.zeros(3), cov=R) for particle in particles]
    
    # weights created with multivariate gaussian because sensor data noise and position noise are gaussian
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    weights = []
    for i in range(len(predict)):
        e = np.array(predict[i] - z_t)
        weight = ((2*np.pi)**(-3/2))*(np.linalg.det(Q)**(-1/2))*np.exp(-0.5 * np.transpose(e) @ np.linalg.inv(Q) @ e)
        weights.append(weight)
    
    weights = np.array(weights)
    if sum(weights != 0):
        weights /= sum(weights)
    else:
        weights = np.zeros_like(weights)

    # X_bar = [(predict[i], weights[i]) for i in range(len(predict))]

    X_indices = np.random.choice(len(predict), len(predict), True, weights)

    X_ = [predict[index] for index in X_indices]
        
    return X_

def PotentialField(pos, goal, detected, field=FIELD, k_att=1.0, k_rep=1000.0):
    k_att = 1.0
    k_rep = 50.0

    attractive_force = -k_att * (pos-goal)
    repulsive_force = np.zeros(2)
    for obstacle in detected:
        d_q = max(np.linalg.norm(obstacle - pos), EPSILON) # prevent divide by zero
        if d_q < field:
            g_dq = (obstacle - pos) / d_q
            repulsive_force += 0.5*k_rep*((1 / field)-(1 / d_q)) * (1 / (d_q**2)) * g_dq

    return attractive_force+repulsive_force

def distance_to_closest_obstacle(pos, obstacles):
    min_dist = 1000
    for obstacle in obstacles:
        ob_dist = np.linalg.norm(pos-obstacle)
        if ob_dist < min_dist:
            min_dist = ob_dist
    return min_dist

# written by copilot 04/18/2026 to implement object filtering within a tolerance
class ClusterMeanSet:
    def __init__(self, tol=1):
        self.tol = tol
        self.clusters = []   # each cluster: {"mean": np.array([x,y]), "count": N}

    def add(self, point):
        point = np.asarray(point)

        # Try to merge with an existing cluster
        for cluster in self.clusters:
            if np.linalg.norm(point - cluster["mean"]) < self.tol:
                # Incremental mean update
                n = cluster["count"]
                cluster["mean"] = cluster["mean"] + (point - cluster["mean"]) / (n + 1)
                cluster["count"] += 1
                return

        # Otherwise create a new cluster
        self.clusters.append({"mean": point.copy(), "count": 1})

    def get_points(self):
        """Return list of cluster means as Nx2 array."""
        return [c["mean"] for c in self.clusters]

    def __iter__(self):
        return iter(self.get_points())

    def __len__(self):
        return len(self.clusters)


class BicycleRobot:
    def __init__(self, name, color, w, h, L, x, y, r, loc_particles, R, Q, detect_range, detect_fov_deg, goal_positions) -> None:
        self.name = name
        self.color = color
        self.width = w
        self.height = h
        self.L = L
        self.x_init = x
        self.y_init = y
        self.r_init = r
        self.true_pos = np.array([x,y,r], dtype=float)
        self.particles_init = loc_particles
        self.particles = loc_particles.copy()
        self.R = R
        self.Q = Q
        self.detection_range = detect_range
        self.detection_fov_rad = np.deg2rad(detect_fov_deg)
        self.detected_obs = ClusterMeanSet()
        self.timer = 0
        self.path_len = 0
        self.error_over_time = []
        self.distance_to_closest_object = []
        self.step = 0
        self.positions = goal_positions.copy()
        self.goal_distance = 1000.0
        self.withinCycleDist = False

    def reset(self):
        self.true_pos = np.array([self.x_init,self.y_init,self.r_init], dtype=float)
        self.detected_obs = ClusterMeanSet()
        self.timer = 0
        self.path_len = 0
        self.error_over_time = []
        self.distance_to_closest_object = []
        self.particles = self.particles_init.copy()
        self.step = 0
        self.goal_distance = 1000.0
        self.withinCycleDist = False

    # shape generation by copilot 04/14/2026
    def get_detector_polygon(self, num_points=30):
        theta = self.true_pos[2]
        x, y = self.get_front_position()

        half_fov = self.detection_fov_rad / 2
        angles = np.linspace(theta - half_fov, theta + half_fov, num_points)

        arc_x = x + self.detection_range * np.cos(angles)
        arc_y = y + self.detection_range * np.sin(angles)

        # Polygon: start at robot, sweep arc, return to robot
        poly_x = np.concatenate([[x], arc_x, [x]])
        poly_y = np.concatenate([[y], arc_y, [y]])

        return poly_x, poly_y

    def get_front_position(self):
        x, y, theta = self.true_pos
        front_x = x + (3 * self.height / 4) * np.cos(theta)
        front_y = y + (3 * self.height / 4) * np.sin(theta)
        return front_x, front_y

    def draw_robot(self, ax):
        width = self.width
        height = self.height
        center_x = self.true_pos[0]
        center_y = self.true_pos[1]
        theta = self.true_pos[2]

        corner1 = [
            center_x - (height / 4) * np.cos(theta) + (width / 2) * np.sin(theta),
            center_y - (height / 4) * np.sin(theta) - (width / 2) * np.cos(theta),
        ]
        corner2 = [
            center_x - (height / 4) * np.cos(theta) - (width / 2) * np.sin(theta),
            center_y - (height / 4) * np.sin(theta) + (width / 2) * np.cos(theta),
        ]
        corner3 = [
            center_x + (3*height / 4) * np.cos(theta) - (width / 2) * np.sin(theta),
            center_y + (3*height / 4) * np.sin(theta) + (width / 2) * np.cos(theta),
        ]
        corner4 = [
            center_x + (3*height / 4) * np.cos(theta) + (width / 2) * np.sin(theta),
            center_y + (3*height / 4) * np.sin(theta) - (width / 2) * np.cos(theta),
        ]

        corners = np.array([corner1, corner2, corner3, corner4, corner1]).T

        front_x, front_y = self.get_front_position()

        # Heading line
        x = [center_x, center_x + height * np.cos(theta)]
        y = [center_y, center_y + height * np.sin(theta)]
        heading_line, = ax.plot(x, y, 'k-')

        # Robot body
        body_line, = ax.plot(corners[0, :], corners[1, :], 'b')

        # --- Draw detector --- by copilot 04/14/2026
        poly_x, poly_y = self.get_detector_polygon()
        detector_patch = ax.fill(poly_x, poly_y, color='orange', alpha=0.2)

        return [heading_line, body_line] + detector_patch
    
    # help writing from copilot 04/14/2026
    def detect(self, obstacles, use_noise=False):
        x, y = self.get_front_position()
        theta = self.true_pos[2]
        detections = []

        for px, py in obstacles:
            mean = np.array([px, py, 0])
            noise = np.zeros(3)
            if use_noise:
                noise = np.random.multivariate_normal(np.zeros(3), cov=self.Q)
            total = mean+noise

            dx, dy = total[0] - x, total[1] - y
            dist = np.hypot(dx, dy)
            if dist > self.detection_range:
                continue

            angle = np.arctan2(dy, dx)
            dtheta = np.arctan2(np.sin(angle - theta), np.cos(angle - theta))

            if abs(dtheta) <= self.detection_fov_rad / 2:
                mean = np.array([px, py, 0])
                noise = np.random.multivariate_normal(np.zeros(3), cov=self.Q)
                total = mean+noise
                detections.append(total[:2]) # ignore heading

        return detections

    def controller(self, dt, obstacles):
        """Runs with timestep dt. Returns True while running and False when complete."""
        if (self.step >= len(self.positions)):
            return False
        
        goal_position = self.positions[self.step]

        mean = np.array(particle_mean(self.particles))
        detected = self.detect(obstacles, True)
        for obj in detected:
            self.detected_obs.add(obj)

        F = PotentialField(mean, goal_position, self.detected_obs, field=self.detection_range*max(self.width, self.height))
        theta_ = np.atan2(F[1], F[0])
        gamma = wrap_to_pi(K_h * (theta_ - self.true_pos[2]))

        v = min(MAX_SPEED, max(K_v*np.linalg.norm(F), -MAX_SPEED)) # bound to max speed
        gamma = np.clip(gamma, -MAX_TURN, MAX_TURN)

        u_t = np.array([v*np.cos(self.true_pos[2])*dt, v*np.sin(self.true_pos[2])*dt, (v/self.L)*np.tan(gamma)*dt])
                        # np.random.multivariate_normal(np.zeros(3), cov=R)
        
        z_t = np.eye(3) @ np.asarray([self.true_pos[0], self.true_pos[1], self.true_pos[2]]) + \
                        np.random.multivariate_normal(np.zeros(3), cov=self.Q)
        
        self.particles = ParticleFilter(self.particles, u_t, z_t, self.R, self.Q)

        self.true_pos += u_t
        self.path_len += np.linalg.norm(u_t[:2])
        self.timer += dt
        self.distance_to_closest_object.append(distance_to_closest_obstacle(self.true_pos[:2], obstacles))
        self.error_over_time.append(np.linalg.norm(mean[:2] - self.true_pos[:2]))

        self.goal_distance = np.linalg.norm(self.true_pos[:2] - goal_position[:2])
        if (self.goal_distance < 1.0):
            self.withinCycleDist = True
        if (self.goal_distance < 0.5 or (self.goal_distance > 1.0 and self.withinCycleDist)):
            self.step += 1
            self.withinCycleDist = False
        return True
