import numpy as np

def distance(point1, point2) -> float:
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

class Robot:
    def __init__(self, x, y, a,):

        self.max_detection_range = 1.5 # m
        self.fov_angle = np.deg2rad(60)
        self.radius = 0.177 # m
        self.linear_speed = 0.2 # m/s
        self.linear_acceleration = 0.4 # m/s^2
        self.max_angular_speed = np.pi / 8 # rad/s
        self.min_angular_speed = np.pi / 16
        self.min_linear_speed = 0.05
        
        self.distance_allowence = 0.08
        self.path_update_distance = 0.04
        self.angular_speed_distance_allowence = 0.06
        self.angle_allowence = np.deg2rad(1)
        self.obstacle_clearence = 0.045 # m
        
        self.x = x
        self.y = y
        self.a = a
    
    def __str__(self) -> str:
        return f'Robot coordinates: {self.x, self.y, self.a}'

class Checkpoint:
    def __init__(self, x, y, a):
        self.x = x
        self.y = y
        self.a = a
    def __str__(self) -> str:
        return f'Checkpoint coordinates: {self.x, self.y, self.a}'

class Obstacle:
    def __init__(self, x, y, color, radius = 0.025):
        self.x: float = x 
        self.y: float = y 
        self.radius = radius 
        self.color: int = color
    def __str__(self) -> str:
        return f'Obstacle coordinates and color: {self.x, self.y, self.color}'
            
class Environment:
    def __init__(self, robot, real_robot, checkpoints, obstacles, hidden_obstacles):
        self.robot: Robot = robot
        self.checkpoints: list[Checkpoint] = checkpoints
        self.obstacles: list[Obstacle] = obstacles
        self.primary_checkpoints_idxs: list[int] = []
        self.real_robot: Robot = real_robot
        
        self.found_finish = False

        self.hidden_obstacles: set[Obstacle] = hidden_obstacles

    def add_checkpoint_for_obstacle_pair(self, obstacle_red, obstacle_blue):
        proximity_threshold = 0.30  # up to  cm is the same checkpoint, probably should be higher
        center = ((obstacle_red.x + obstacle_blue.x) / 2, (obstacle_red.y + obstacle_blue.y) / 2)
        direction = (obstacle_blue.x - obstacle_red.x, obstacle_blue.y - obstacle_red.y)
        normal = (-direction[1], direction[0])
        side = normal[0] * (self.robot.x - obstacle_red.x) + normal[1] * (self.robot.y - obstacle_red.y)

        if side > 0:
            checkpoint_angle = np.arctan2(normal[1], normal[0])
            turned_checkpoint_angle = checkpoint_angle + np.pi / 2
        else:
            checkpoint_angle = np.arctan2(-normal[1], -normal[0])
            turned_checkpoint_angle = checkpoint_angle - np.pi / 2

        offset_distance = 0.05 + self.robot.radius + obstacle_red.radius + 0.05
        new_checkpoint = Checkpoint(center[0] + offset_distance * np.cos(checkpoint_angle),
                                    center[1] + offset_distance * np.sin(checkpoint_angle),
                                    np.arctan2(np.sin(checkpoint_angle + np.pi), np.cos(checkpoint_angle + np.pi)))
        
        possible_new_checkpoints = Checkpoint(center[0] + offset_distance * np.cos(checkpoint_angle + np.pi),
                                              center[1] + offset_distance * np.sin(checkpoint_angle + np.pi),
                                              np.arctan2(np.sin(checkpoint_angle), np.cos(checkpoint_angle)))

        turned_checkpoint_angle = np.arctan2(np.sin(turned_checkpoint_angle), np.cos(turned_checkpoint_angle))
        turned_checkpoint = Checkpoint(new_checkpoint.x, new_checkpoint.y, turned_checkpoint_angle)

        updated = False
        for idx in self.primary_checkpoints_idxs:
            if distance(new_checkpoint, self.checkpoints[idx]) < proximity_threshold:
                self.checkpoints[idx] = new_checkpoint
                self.checkpoints[idx + 1] = turned_checkpoint
                updated = True
                break
            # already visited checkpoint but the robot got behind the obstacles
            if distance(self.checkpoints[idx], possible_new_checkpoints) < proximity_threshold:
                updated = True
                break

        if not updated and not self.found_finish:
            self.checkpoints.append(new_checkpoint)
            self.primary_checkpoints_idxs.append(len(self.checkpoints) - 1)
            self.checkpoints.append(turned_checkpoint)

            return True
        return False

    def add_finish(self, obstacle_one, obstacle_two):
        proximity_threshold = 0.20  # up to 20 cm is the same checkpoint, probably should be higher
        center = ((obstacle_one.x + obstacle_two.x) / 2, (obstacle_one.y + obstacle_two.y) / 2)
        direction = (obstacle_two.x - obstacle_one.x, obstacle_two.y - obstacle_one.y)
        normal = (-direction[1], direction[0])
        side = normal[0] * (self.robot.x - obstacle_one.x) + normal[1] * (self.robot.y - obstacle_one.y)

        if side > 0:
            checkpoint_angle = np.arctan2(normal[1], normal[0])
        else:
            checkpoint_angle = np.arctan2(-normal[1], -normal[0])

        offset_distance = 0.05 + self.robot.radius + obstacle_one.radius + 0.05
        new_x = center[0] + offset_distance * np.cos(checkpoint_angle)
        new_y = center[1] + offset_distance * np.sin(checkpoint_angle)
        checkpoint_angle = np.arctan2(np.sin(checkpoint_angle + np.pi), np.cos(checkpoint_angle + np.pi))

        # Check if there's an existing checkpoint nearby
        updated = False
        # for checkpoint in self.checkpoints:
        # for idx in self.primary_checkpoints_idxs:
        if np.sqrt((self.checkpoints[-1].x - new_x)**2 + (self.checkpoints[-1].y - new_y)**2) < proximity_threshold:
            # Adjust existing self.checkpoints[-1] position and angle
            self.checkpoints[-1].x = new_x
            self.checkpoints[-1].y = new_y
            self.checkpoints[-1].a = checkpoint_angle
            updated = True
            self.found_finish = True
            return

        if not updated:
            # Create new checkpoint
            new_checkpoint = Checkpoint(new_x, new_y, checkpoint_angle)
            self.checkpoints.append(new_checkpoint)
            self.found_finish = True
            print("--------------- Found FINISH ---------------")
        return

    def update_checkpoints(self):
        obstacles_red = [obstacle for obstacle in self.obstacles if obstacle.color == 0]
        obstacles_blue = [obstacle for obstacle in self.obstacles if obstacle.color == 1]
        obstacles_green = [obstacle for obstacle in self.obstacles if obstacle.color == 2]
        added_new = False
        for obstacle_red in obstacles_red:
            for obstacle_blue in obstacles_blue:
                if abs(distance(obstacle_blue, obstacle_red) - 0.055) < 0.1:
                    if self.add_checkpoint_for_obstacle_pair(obstacle_red, obstacle_blue):
                        added_new = True

        for i in range(len(obstacles_green)):
            for j in range(len(obstacles_green)):
                if i == j:
                    continue
                if abs(distance(obstacles_green[i], obstacles_green[j]) - 0.055) < 0.1:
                    self.add_finish(obstacles_green[i], obstacles_green[j])
        return added_new                    
                        
    def simulate_movement(self, move, time):
        # Introduce noise in linear and angular velocities
        linear_noise = 0 * np.random.normal(0, 0.05)  # Mean 0, standard deviation 0.05
        angular_noise = 0 * np.random.normal(0, 0.1) # Mean 0, standard deviation 0.01

        # Apply the noise to the movement values
        noisy_linear = move[0] + linear_noise
        noisy_angular = move[1] + angular_noise

        measured_delta_x = np.cos(self.real_robot.a) * move[0] * time
        measured_delta_y = np.sin(self.real_robot.a) * move[0] * time
        measured_delta_theta = move[1] * time

        noisy_delta_x = np.cos(self.real_robot.a) * noisy_linear * time
        noisy_delta_y = np.sin(self.real_robot.a) * noisy_linear * time
        noisy_delta_theta = noisy_angular * time
        # Update the robot's position and orientation with the noisy values
        self.real_robot.x += measured_delta_x
        self.real_robot.y += measured_delta_y
        self.real_robot.a += measured_delta_theta
        
        # Normalize the angle
        self.real_robot.a = np.arctan2(np.sin(self.real_robot.a), np.cos(self.real_robot.a))
        
        return (noisy_delta_x, noisy_delta_y, noisy_delta_theta)

    def get_measurement(self):
        measurements = []
        for obstacle in self.hidden_obstacles:
            # Calculate true distance and angle to the obstacle
            true_distance_to_obstacle = distance(self.real_robot, obstacle)
            true_angle_to_obstacle = np.arctan2(obstacle.y - self.real_robot.y, obstacle.x - self.real_robot.x) - self.real_robot.a

            # Check if the obstacle is within the detectable range and field of view
            if (true_distance_to_obstacle <= self.real_robot.max_detection_range and
                    -self.real_robot.fov_angle / 2 <= true_angle_to_obstacle <= self.real_robot.fov_angle / 2):
                
                # Add noise to distance and angle measurements
                distance_noise = 0 * np.random.normal(0, 0.03)
                angle_noise = 0 * np.random.normal(0, np.radians(1))

                # Ensure that the measurements are scalar
                measured_distance = min(self.real_robot.max_detection_range, float(true_distance_to_obstacle + distance_noise))
                measured_angle = min(self.real_robot.fov_angle, float(true_angle_to_obstacle + angle_noise))

                measurements.append([measured_distance, measured_angle, float(obstacle.color)])
                
        return measurements
    
    def get_odometry(self):
        return (self.real_robot.x, self.real_robot.y, self.real_robot.a)
    