import math
from a_star import A_star
from path_execution import PathExecution

def distance(point1, point2) -> float:
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

class Robot:
    def __init__(self,
            x, y, a,
            radius = 17.7, # cm
            linear_speed = 0.2, # m/s
            angular_speed = math.pi / 6, # rad/s
            obstacle_clearence = 3, # cm
        ):

        self.radius = radius
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.obstacle_clearence = obstacle_clearence

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
        self.checkpoints: list[Checkpoint] = checkpoints
        self.obstacles: set[Obstacle] = obstacles
        
        self.current_goal_checkpoint_index = 0
        self.path = []
        
        self.path_execution = PathExecution(self)
        
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
                    came_from, cost_so_far = a_star.search(goal)
                    if goal in came_from:
                        self.path = self.reconstruct_path(came_from, goal)
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
        changes = (math.cos(math.radians(self.robot.a)) * move[0] * time,
                    math.sin(math.radians(self.robot.a)) * move[0] * time,
                    move[1] * time)
        self.robot.x += changes[0] * 100
        self.robot.y += changes[1] * 100
        self.robot.a += (math.degrees(changes[2]) + 180) % 360 - 180
    
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

    def in_collision(self, node) -> bool:
        for obstacle in self.obstacles:
            allowed_distance = obstacle.radius + self.robot.radius + self.robot.obstacle_clearence
            if distance(node, obstacle) < allowed_distance:
                return True
        return False

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

