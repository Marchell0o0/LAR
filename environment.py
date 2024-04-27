import math
import numpy as np

class Node:
    def __init__(self, x, y, a = 0):
        self.x = x
        self.y = y
        self.a = a

    def __lt__(self, other):
        # Less than (<) comparison, could be based on heuristic, cost, etc.
        return (self.x, self.y, self.a) < (other.x, other.y, other.a)

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return (self.x == other.x) and (self.y == other.y) and (self.a == other.a)

    def __hash__(self):
        return hash((self.x, self.y, self.a))

def distance(point1: Node, point2: Node) -> float:
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


class Robot:
    def __init__(self,
            x, y, a,
            radius = 17.7, # cm
            linear_speed = 0.2, # m/s
            angular_speed = math.pi / 6, # rad/s
            obstacle_clearence = 3, # cm
            viewing_distance = 20 # cm
        ):

        self.radius = radius
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.obstacle_clearence = obstacle_clearence

        # Should be removed in the final version
        # coordinate system will always have the robot at (0, 0, 0)
        self.x = x
        self.y = y
        self.a = a

class Checkpoint:
    def __init__(self, x, y, a):
        self.x = x
        self.y = y
        self.a = a

class Obstacle:
    def __init__(self, x, y, radius = 2.5):
        self.x = x 
        self.y = y 
        self.radius = radius 

class Environment:
    def __init__(self, robot, checkpoints, obstacles):
        self.robot: Robot = robot
        self.checkpoints: List[Checkpoint] = checkpoints
        self.obstacles: Set[Obstacle] = obstacles
        for obstacle in list(self.obstacles):  # Create a copy of the set for iteration
            if distance(obstacle, robot) < (robot.radius 
                                            + obstacle.radius
                                            + robot.obstacle_clearence):
                self.obstacles.remove(obstacle)


    def simulate_movement(self, move, time, odometry):
        """

        Args:
            move (tuple[int, int]): values of linear and angular velocities m/s, rad/s
            time (float): time to move
        """
        changes = (math.cos(math.radians(self.robot.a)) * move[0] * time,
                    math.sin(math.radians(self.robot.a)) * move[0] * time,
                    move[1] * time)
        self.robot.x += changes[0] * 100
        self.robot.y += changes[1] * 100
        self.robot.a += math.degrees(changes[2])
        new_odometry = (odometry[0] + changes[0],
                        odometry[1] + changes[1],
                        odometry[2] + changes[2])
        return new_odometry
    
    def set_obstacles_from_checkpoints(self):
        for check in self.checkpoints:            
            sin = math.sin(math.radians(check.a))   
            cos = math.cos(math.radians(check.a))
            
            # 5 cm to the sides
            move_sideways_cos = 5 * cos 
            move_sideways_sin = 5 * sin

            # robot should be 5cm from the obstacle
            move_away_cos = (self.robot.radius + 5) * cos 
            move_away_sin = (self.robot.radius + 5) * sin

            right_x = check.x - move_sideways_sin + move_away_cos
            right_y = check.y + move_sideways_cos + move_away_sin

            left_x = check.x + move_sideways_sin + move_away_cos
            left_y = check.y - move_sideways_cos + move_away_sin

            self.obstacles.add(Obstacle(left_x, left_y))
            self.obstacles.add(Obstacle(right_x, right_y))
        return

    def in_collision(self, node: Node) -> bool:
        for obstacle in self.obstacles:
            allowed_distance = obstacle.radius + self.robot.radius + self.robot.obstacle_clearence
            if distance(node, obstacle) < allowed_distance:
                return True
        return False


    def straight_path_exists(self, start, goal) -> bool:
        # print("Looking for straight path...")
        if start.x == goal.x and start.y == goal.y:
            return True
        # check for obstacle collisions
        for obstacle in self.obstacles:
            # Transform system to obstacle center
            robot_x = start.x - obstacle.x
            robot_y = start.y - obstacle.y
            goal_x = goal.x - obstacle.x
            goal_y = goal.y - obstacle.y

            # The combined radius of the robot and obstacle with clearance
            radius = obstacle.radius + self.robot.radius + self.robot.obstacle_clearence

            # Coefficients for the quadratic equation
            dx = goal_x - robot_x
            dy = goal_y - robot_y
            a = dx**2 + dy**2
            b = 2 * (robot_x * dx + robot_y * dy)
            c = robot_x**2 + robot_y**2 - radius**2

            # Discriminant
            d = b**2 - 4 * a * c
            if d < 0:
                # No intersection
                continue

            # Intersection(s) exist
            dsqrt = math.sqrt(d)
            t1 = (-b - dsqrt) / (2 * a)
            t2 = (-b + dsqrt) / (2 * a)
            if (0 < t1 < 1) or (0 < t2 < 1):
                # print(f"Collision with obstacle at ({obstacle.x}, {obstacle.y})")
                return False
            
        # No collisions detected
        return True

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