# import np
import numpy as np
from a_star import A_star
from path_execution import PathExecution
# from kalman import KalmanFilter


def distance(point1, point2) -> float:
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

class Robot:
    def __init__(self, x, y, a,):

        self.max_detection_range = 0.7 # m
        self.fov_angle = np.deg2rad(90)
        self.radius = 0.177 # m
        self.linear_speed = 1 # m/s
        self.angular_speed = np.pi / 3 # rad/s
        self.obstacle_clearence = 0.03 # m

        self.node_distance_allowance = 0.01 # m
        self.node_angle_allowance = np.deg2rad(1) # deg
        self.linear_acceleration_time = 0.1 # s TODO: WRONG
        self.angular_acceleration_time = 0.1 # s TODO: WRONG
        self.minimal_linear_speed = 0.05 # m/s
        self.minimal_angular_speed = 0.3 # rad/s

        self.x = x
        self.y = y
        self.a = a

class Checkpoint:
    def __init__(self, x, y, a):
        self.x = x
        self.y = y
        self.a = a

class Obstacle:
    def __init__(self, x, y, color, radius = 0.025):
        self.x = x 
        self.y = y 
        self.radius = radius 
        self.color = color
            
class Environment:
    def __init__(self, robot, checkpoints, obstacles, hidden_obstacles):
        self.robot: Robot = robot
        self.checkpoints: list[Checkpoint] = checkpoints
        self.obstacles: list[Obstacle] = obstacles
        
        self.hidden_obstacles: set[Obstacle] = hidden_obstacles
        
        self.current_goal_checkpoint_index = 0
        self.path = []
        
        self.path_execution = PathExecution(self)
        
        self.kalman_filter = KalmanFilter(self)
        
        for obstacle in self.obstacles:
            if distance(obstacle, robot) < (robot.radius 
                                            + obstacle.radius
                                            + robot.obstacle_clearence):
                print("Obstacle is already too close to the robot")


    def reconstruct_path(self, came_from, goal):
        current = goal
        path = []
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def reset_path(self):
        self.path = []
        

    def get_current_move(self):
        if not self.path:
            # Still have checkpoints to go through
            if self.current_goal_checkpoint_index < len(self.checkpoints):                
                goal = self.checkpoints[self.current_goal_checkpoint_index]
                if self.straight_path_exists(self.robot, goal):
                    print("Found straight path")
                    self.path = [goal]
                    self.path_execution.update()
                else:
                    print("Looking for path with A*")
                    a_star = A_star(self)
                    self.path = a_star.search(goal)
                    if self.path:
                        self.path = self.simplify_path(self.path)
                        self.path.pop(0) # receiving the path with robot at start 
                        self.path_execution.update()
                        print("Path found using A*")
                    else:
                        print("Couldn't find path with A*")
                        return (0, 0)
            # Finished executing current checkpoints (looking for other ones)
            else:
                return (0, 0)
        # Here the path is guaranteed to be non-empty
        return self.path_execution.move_through_path()

    def simulate_movement(self, move, time):
        self.robot.x += np.cos(self.robot.a) * move[0] * time
        self.robot.y += np.sin(self.robot.a) * move[0] * time
        self.robot.a += move[1] * time
        self.robot.a = np.arctan2(np.sin(self.robot.a), np.cos(self.robot.a))
    
    import numpy as np

    def get_measurement(self):
        measurements = []
        for obstacle in self.hidden_obstacles:
            true_distance_to_obstacle = distance(self.robot, obstacle)
            true_angle_to_obstacle = np.arctan2(obstacle.y - self.robot.y, obstacle.x - self.robot.x) - self.robot.a

            # Check if the obstacle is within the detectable range and field of view
            if (true_distance_to_obstacle <= self.robot.max_detection_range and
                    -self.robot.fov_angle / 2 <= true_angle_to_obstacle <= self.robot.fov_angle / 2):
                
                # Add noise to distance and angle measurements
                # Assuming noise mean = 0, and you can adjust the standard deviation (std_dev) to represent sensor accuracy
                distance_noise = np.random.normal(0, 0.05 * true_distance_to_obstacle)  # 5% noise relative to the distance
                angle_noise = np.random.normal(0, np.radians(2))  # 5 degrees noise

                measured_distance = true_distance_to_obstacle + distance_noise
                measured_angle = true_angle_to_obstacle + angle_noise

                measurements.append([measured_distance, measured_angle, obstacle.color, obstacle.index])
                # Optionally add the obstacle to the detected set if processing immediately
                # self.obstacles.add(obstacle)

        return measurements

    
    # def get_measurement(self):
    #     measurements = []
    #     for obstacle in self.hidden_obstacles:
    #         distance_to_obstacle = distance(self.robot, obstacle)

    #         angle_to_obstacle = np.arctan2(obstacle.y - self.robot.y, obstacle.x - self.robot.x) - self.robot.a

    #         if (distance_to_obstacle <= self.robot.max_detection_range and
    #                 -self.robot.fov_angle / 2 <= angle_to_obstacle <= self.robot.fov_angle / 2):
    #             measurements.append([distance_to_obstacle, angle_to_obstacle, obstacle.color, obstacle.index])
    #             # self.obstacles.add(obstacle)
    #     return measurements

    def simplify_path(self, path):
        simplified_path = [path[0]]  # Start with the first point
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self.straight_path_exists(path[i], path[j]):
                    # Found a direct line to a further point
                    i = j
                    break
                j -= 1
            simplified_path.append(path[i])
            i += 1
        return simplified_path
    
    def straight_path_exists(self, start, goal) -> bool:
        if start.x == goal.x and start.y == goal.y:
            return True
        for obstacle in self.obstacles:
            # Transform system to obstacle center
            robot_x = start.x - obstacle.x
            robot_y = start.y - obstacle.y
            goal_x = goal.x - obstacle.x
            goal_y = goal.y - obstacle.y

            radius = obstacle.radius + self.robot.radius + self.robot.obstacle_clearence

            dx = goal_x - robot_x
            dy = goal_y - robot_y
            a = dx**2 + dy**2
            b = 2 * (robot_x * dx + robot_y * dy)
            c = robot_x**2 + robot_y**2 - radius**2
            d = b**2 - 4 * a * c
            if d < 0:
                # No intersection
                continue

            # Intersection(s) exist
            dsqrt = np.sqrt(d)
            t1 = (-b - dsqrt) / (2 * a)
            t2 = (-b + dsqrt) / (2 * a)
            if (0 < t1 < 1) or (0 < t2 < 1):
                return False
            
        return True




class KalmanFilter:
    def __init__(self, env):
        self.env = env
        self.sigma = 0.01*np.eye(3, dtype=float)
        self.R = np.diag([0.0005, 0.0005, 0.0005]) # sigma x, y, a
        self.Q = np.diag([0.1, np.deg2rad(2)]) # sigma r, phi
        
        self.mu = np.zeros((3, 1))

    def Fx(self):
        total_state_length = 3 + 2 * len(self.env.obstacles)  # 3 for robot state, 2 per obstacle
        Fx_base = np.eye(3)  # Base for robot state
        Fx_obstacles = np.zeros((3, total_state_length - 3))  # Extend to cover all obstacles
        return np.block([Fx_base, Fx_obstacles])  # Combined state transition matrix

    def prediction_update(self, u, dt):
        theta = self.env.robot.a
        v, w = u[0], u[1]
        # Update state estimate mu with model
        state_model_mat = np.zeros((3, 1)) # Initialize state update matrix from model
        state_model_mat[0] = -(v/w)*np.sin(theta)+(v/w)*np.sin(theta+w*dt) if w>0.01 else v*np.cos(theta)*dt
        state_model_mat[1] = (v/w)*np.cos(theta)-(v/w)*np.cos(theta+w*dt) if w>0.01 else v*np.sin(theta)*dt 
        state_model_mat[2] = w*dt
        
        # self.env.robot.x +=  state_model_mat[0]                                                    
        # self.env.robot.y +=  state_model_mat[1]                                                    
        # self.env.robot.a +=  state_model_mat[2]        
        self.mu[0:3] += state_model_mat           
                                                    
                                                    
        # Update state uncertainty sigma
        state_jacobian = np.zeros((3,3)) # Initialize model jacobian
        state_jacobian[0,2] = (v/w)*np.cos(theta) - (v/w)*np.cos(theta+w*dt) if w>0.01 else -v*np.sin(theta)*dt # Jacobian element, how small changes in robot theta affect robot x
        state_jacobian[1,2] = (v/w)*np.sin(theta) - (v/w)*np.sin(theta+w*dt) if w>0.01 else v*np.cos(theta)*dt # Jacobian element, how small changes in robot theta affect robot y
        
        # print("Fx shape:", Fx.shape)
        # print("state_jacobian shape:", state_jacobian.shape)
        # print("sigma shape before update:", self.sigma.shape)
        # print("sigma shape after update:", self.sigma.shape)
        # print("G shape:", G.shape)
        
        Fx = self.Fx()
        G = np.eye(self.sigma.shape[0]) + np.transpose(Fx).dot(state_jacobian).dot(Fx)
        self.sigma = G.dot(self.sigma).dot(np.transpose(G)) + np.transpose(Fx).dot(self.R).dot(Fx)
        return


    def measurement_update(self, zs):
        for z in zs:
            dist, phi, color = z
            measured_x = self.mu[0] + dist * np.cos(phi + self.mu[2])
            measured_y = self.mu[1] + dist * np.sin(phi + self.mu[2])
            measured_obstacle = Obstacle(measured_x, measured_y, color)
            # Calculate where the obstacle should be in the state vector
            # idx = 3 + 2 * index  # Assuming robot state (3) + all previous obstacles

            # Check if we need to expand the state vector and covariance matrix
            required_size = idx + 2  # +2 for x and y of new obstacle
            current_size = self.mu.shape[0]
            
            if required_size > current_size:
                # Expand the state vector
                extension = np.zeros((required_size - current_size, 1))
                self.mu = np.vstack((self.mu, extension))
                
                # Expand the covariance matrix
                sigma_extension = np.zeros((required_size, required_size))
                sigma_extension[:current_size, :current_size] = self.sigma
                new_diag = 1 * np.eye(required_size - current_size)
                sigma_extension[current_size:, current_size:] = new_diag
                self.sigma = sigma_extension

            # Update or initialize obstacle coordinates
            if index >= len(self.env.obstacles):
                # New obstacle detected, initialize its state
                x_obstacle = self.env.robot.x + dist * np.cos(phi + self.env.robot.a)
                y_obstacle = self.env.robot.y + dist * np.sin(phi + self.env.robot.a)
                obstacle = Obstacle(x_obstacle, y_obstacle, color, index)
                self.env.obstacles.add(obstacle)
                self.mu[idx] = x_obstacle
                self.mu[idx + 1] = y_obstacle
            else:
                for obst in self.env.obstacles:
                    if obst.index == index:
                        obstacle = obst

            # Calculate the expected measurement and Jacobian
            delta = np.array([[obstacle.x - self.env.robot.x], [obstacle.y - self.env.robot.y]])
            q = np.linalg.norm(delta)**2
            # z_hat = np.array([[np.sqrt(q)], [np.arctan2(delta[1, 0], delta[0, 0]) - self.env.robot.a]])
            sqrt_q = np.sqrt(q)
            angle = np.arctan2(delta[1, 0], delta[0, 0]) - self.env.robot.a
            z_hat = np.vstack((sqrt_q, angle)).reshape(2, 1)

            H = self.calculate_jacobian(delta, q, idx)
            S = H @ self.sigma @ H.T + self.Q
            K = self.sigma @ H.T @ np.linalg.inv(S)

            z_actual = np.array([[dist], [phi]])
            z_actual = np.reshape(z_actual, (2, 1))
            # Apply the Kalman gain to the whole state vector
            self.mu += K @ (z_actual - z_hat)

            # Update the entire covariance matrix
            self.sigma = (np.eye(self.sigma.shape[0]) - K @ H) @ self.sigma

    def calculate_jacobian(self, delta, q, idx):
        # Compute the Jacobian matrix for the measurement function
        H = np.zeros((2, self.sigma.shape[0]))
        H[0, 0] = -delta[0] / np.sqrt(q)
        H[0, 1] = -delta[1] / np.sqrt(q)
        H[1, 0] = delta[1] / q
        H[1, 1] = -delta[0] / q
        H[0, idx] = delta[0] / np.sqrt(q)
        H[0, idx+1] = delta[1] / np.sqrt(q)
        H[1, idx] = -delta[1] / q
        H[1, idx+1] = delta[0] / q
        return H

    def update_environment(self):
        self.env.robot.x = self.mu[0]
        self.env.robot.y = self.mu[1]
        self.env.robot.a = self.mu[2]
        for idx, obstacle in enumerate(self.env.obstacles):
            mu_idx = 3 + idx * 3
            obstacle.x = self.mu[mu_idx]
            obstacle.y = self.mu[mu_idx + 1]
            obstacle.color = self.mu[mu_idx + 2]
            
    def process_measurement(self, u, z, dt):
        # print("Processing move:", u)
        self.prediction_update(u, dt)
        
        # print("Robot state post-prediction:", self.env.robot.x, self.env.robot.y, self.env.robot.a)
        # print("Processing measurement:", z)
        self.measurement_update(z)
        self.update_environment()
        # print("Robot state post-correction:", self.env.robot.x, self.env.robot.y, self.env.robot.a)

