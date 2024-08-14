from PathManagement.astar import AStar, Node
import numpy as np
from math import comb

def distance(point1: Node, point2: Node):
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


class PathCreation:
    def __init__(self, env) -> None:
        self.env = env
        self.a_star = AStar(env)

    def create_path(self, robot, goal_checkpoint):
        print("--------------- Creating path ---------------")
        # start_time = time.time()
        path = self.initialize_path(robot, goal_checkpoint)
        # init_time = time.time() - start_time
        if not path:
            return path

        path[0] = Node(robot.x, robot.y)
        path[-1] = goal_checkpoint
        if len(path) == 2:
            path = self.inject_nodes(path, 0.02, False)
        else:
            path = self.simplify_path(path, 1, 'start')
            path = self.inject_nodes(path, 0.05, False)
            path = self.simplify_path(path, 1, 'end')
            path = self.inject_nodes(path, 0.02, False)
            path = self.bezier_curve_interpolation(path, 0.02)
        print("---------------------------------------------")
        return path

    def initialize_path(self, robot, goal_checkpoint):
        start = Node(robot.x, robot.y)
        goal = Node(goal_checkpoint.x, goal_checkpoint.y)
        print("Straight path: ", end='')
        if self.straight_path_exists(start, goal_checkpoint):
            print("NO COLLISION")
            return [start, goal]
        print("COLLISION")

        print("A*:        ", end='')
        path = self.a_star.search(goal)
        if path:
            print("    FOUND")
            return path
        print("NOT FOUND")
        return path

    def simplify_path(self, path, threshold, start_side='start'):
        if len(path) == 2:
            return path

        simplified_path = []

        if start_side == 'start':
            simplified_path = [path[0]]  # Start with the first point
            i = 0
            while i < len(path) - 1:
                j = len(path) - 1
                while j > i + 1:
                    if self.straight_path_exists(path[i], path[j]):
                        if distance(path[i], path[j]) <= threshold:
                            # Found a direct line to a further point within the threshold
                            i = j
                            break
                    j -= 1
                simplified_path.append(path[i])
                i += 1
        elif start_side == 'end':
            simplified_path = [path[-1]]  # Start with the last point
            i = len(path) - 1
            while i > 0:
                j = 0
                while j < i - 1:
                    if self.straight_path_exists(path[i], path[j]):
                        if distance(path[i], path[j]) <= threshold:
                            # Found a direct line to a further point within the threshold
                            i = j
                            break
                    j += 1
                simplified_path.append(path[i])
                i -= 1

            # Reverse the path to maintain the correct order from start to end
            simplified_path.reverse()

        # Ensure the last point is included if starting from the beginning
        if simplified_path[-1] != path[-1]:
            simplified_path.append(path[-1])

        # Ensure the first point is included if starting from the end
        if simplified_path[0] != path[0]:
            simplified_path.insert(0, path[0])

        return simplified_path

    @staticmethod
    def inject_nodes(path, frequency, add_more=False):
        new_path = [path[0]]

        for i in range(len(path) - 1):
            start_node = path[i]
            end_node = path[i + 1]
            segment_length = distance(start_node, end_node)

            # Calculate number of interpolated points needed
            num_points = int(segment_length // frequency)
            # Linear interpolation between start_node and end_node
            for j in range(num_points):
                t = j / num_points
                if j == 0 and i == 0:
                    continue
                interpolated_x = start_node.x + t * (end_node.x - start_node.x)
                interpolated_y = start_node.y + t * (end_node.y - start_node.y)
                new_path.append(Node(interpolated_x, interpolated_y))

            # Add the original end_node to the path
            new_path.append(end_node)
        # if add_more and len(new_path) >= 2:
        #     projection_angle = np.arctan2(new_path[-1].y - new_path[-2].y, new_path[-1].x - new_path[-2].x)
        #     new_path.append(Node(new_path[-1].x + frequency * np.cos(projection_angle),
        #                         new_path[-1].y + frequency * np.sin(projection_angle)))

        return new_path


    @staticmethod
    def calculate_path_length(path):
        length = 0.0
        for i in range(1, len(path)):
            length += np.sqrt((path[i].x - path[i - 1].x) ** 2 + (path[i].y - path[i - 1].y) ** 2)
        return length

    def bezier_curve_interpolation(self, path, frequency):
        def bezier_curve(control_points, num_points):
            t = np.linspace(0, 1, num_points)
            n = len(control_points) - 1
            curve = np.zeros((num_points, 2))
            for i in range(num_points):
                point = np.zeros(2)
                for k in range(n + 1):
                    binomial_coefficients = comb(n, k)
                    term = binomial_coefficients * (t[i] ** k) * ((1 - t[i]) ** (n - k)) * np.array(
                        [control_points[k].x, control_points[k].y])
                    point += term.reshape(2)
                curve[i] = point
            return [Node(x, y) for x, y in curve]

        total_length = self.calculate_path_length(path)
        num_points = int(total_length // frequency)

        smooth_path = bezier_curve(path, num_points)
        return smooth_path


    def straight_path_exists(self, start, goal) -> bool:
        dx = goal.x - start.x
        dy = goal.y - start.y
        if dx == 0 and dy == 0:
            return True
        for obstacle in self.env.obstacles:
            # Calculate relative positions to the obstacle
            ox = obstacle.x - start.x
            oy = obstacle.y - start.y

            # Compute the projection of the obstacle center onto the path
            projection = (ox * dx + oy * dy) / (dx * dx + dy * dy)

            # If projection is outside the [0,1] range, skip as it's outside the segment
            if projection < 0 or projection > 1:
                continue

            # Find the closest point on the line segment to the obstacle
            closest_x = start.x + projection * dx
            closest_y = start.y + projection * dy

            # Calculate the distance from the obstacle center to the closest point
            dist_sq = (closest_x - obstacle.x) ** 2 + (closest_y - obstacle.y) ** 2

            # Check if the distance is within the radius (squared)
            clearance = obstacle.radius + self.env.robot.radius + self.env.robot.obstacle_clearance
            if dist_sq < clearance ** 2:
                return False

        return True
