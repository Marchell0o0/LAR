import numpy as np

def distance(point1, point2) -> float:
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

class Robot:
    def __init__(self, x, y, a,):

        self.max_detection_range = 1 # m
        self.fov_angle = np.deg2rad(70)
        self.radius = 0.177 # m
        self.linear_speed = 0.4 # m/s
        # self.linear_speed = 0.4 # m/s
        self.linear_acceleration = 0.1 # m/s^2
        self.max_angular_speed = np.pi / 6 # rad/s
        self.minimal_angular_speed = 0.1
        self.minimal_linear_speed = 0.05
        
        self.obstacle_clearence = 0.05 # m

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
    def __init__(self, robot, checkpoints, obstacles, hidden_obstacles):
        self.robot: Robot = robot
        self.checkpoints: list[Checkpoint] = checkpoints
        self.obstacles: list[Obstacle] = obstacles
        
        self.hidden_obstacles: set[Obstacle] = hidden_obstacles

    def simulate_movement(self, move, time):
        self.robot.x += np.cos(self.robot.a) * move[0] * time
        self.robot.y += np.sin(self.robot.a) * move[0] * time
        self.robot.a += move[1] * time
        self.robot.a = np.arctan2(np.sin(self.robot.a), np.cos(self.robot.a))
    
    def get_measurement(self):
        measurements = []
        for obstacle in self.hidden_obstacles:
            # Calculate true distance and angle to the obstacle
            true_distance_to_obstacle = distance(self.robot, obstacle)
            true_angle_to_obstacle = np.arctan2(obstacle.y - self.robot.y, obstacle.x - self.robot.x) - self.robot.a

            # Check if the obstacle is within the detectable range and field of view
            if (true_distance_to_obstacle <= self.robot.max_detection_range and
                    -self.robot.fov_angle / 2 <= true_angle_to_obstacle <= self.robot.fov_angle / 2):
                
                # Add noise to distance and angle measurements
                distance_noise = np.random.normal(0, 0.01 * true_distance_to_obstacle)
                angle_noise = np.random.normal(0, np.radians(1))

                # Ensure that the measurements are scalar
                measured_distance = min(self.robot.max_detection_range, float(true_distance_to_obstacle + distance_noise))
                measured_angle = min(self.robot.fov_angle, float(true_angle_to_obstacle + angle_noise))

                measurements.append([measured_distance, measured_angle, float(obstacle.color)])
                
        return measurements

