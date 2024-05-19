from PathManagement.astar import AStar, Node
import numpy as np
import time


def distance(point1: Node, point2: Node):
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


class PathCreation:
    def __init__(self, env) -> None:
        self.env = env
        self.a_star = AStar(env)
        self.node_frequency_on_injection = 0.02
        self.smoother_weight_smooth = 0.9
        self.smoother_weight_data = 1 - self.smoother_weight_smooth
        self.smoother_tolerance = 0.001

    def create_path(self, robot, goal_checkpoint):
        print("--------------- Creating path ---------------")
        start_time = time.time()
        path = self.initialize_path(robot, goal_checkpoint)
        if not path:
            return path

        path[0] = Node(robot.x, robot.y)
        path[-1] = goal_checkpoint

        init_time = time.time() - start_time

        start_time = time.time()
        path = self.simplify_path(path)
        simplify_time = time.time() - start_time

        start_time = time.time()
        path = self.inject_nodes(path)
        inject_time = time.time() - start_time

        # print(f"Time to initialize path: {init_time:.4f} seconds")
        # print(f"Time to simplify path: {simplify_time:.4f} seconds")
        # print(f"Time to inject nodes: {inject_time:.4f} seconds")

        path = self.smoother(path)

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

    def simplify_path(self, path):
        if len(path) == 2:
            return path
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

        # Ensure the last point is included
        if simplified_path[-1] != path[-1]:
            simplified_path.append(path[-1])

        return simplified_path

    def inject_nodes(self, path):
        new_path = []

        for i in range(len(path) - 1):
            start_node = path[i]
            end_node = path[i + 1]
            segment_length = distance(start_node, end_node)

            # Calculate number of interpolated points needed
            num_points = int(segment_length // self.node_frequency_on_injection)

            # Linear interpolation between start_node and end_node
            for j in range(num_points):
                t = j / num_points
                interpolated_x = start_node.x + t * (end_node.x - start_node.x)
                interpolated_y = start_node.y + t * (end_node.y - start_node.y)
                new_path.append(Node(interpolated_x, interpolated_y))

            # Add the original end_node to the path
            new_path.append(end_node)

        if len(new_path) >= 2:
            projection_angle = np.arctan2(new_path[-1].y - new_path[-2].y, new_path[-1].x - new_path[-2].x)
            new_path.append(Node(new_path[-1].x + self.node_frequency_on_injection * np.cos(projection_angle),
                                new_path[-1].y + self.node_frequency_on_injection * np.sin(projection_angle)))

        return new_path

    def smoother(self, path):
        # Create a deep copy of the path to avoid modifying the original
        new_path = [Node(node.x, node.y) for node in path]
        tolerance = self.smoother_tolerance
        weight_data = self.smoother_weight_data
        weight_smooth = self.smoother_weight_smooth

        change = tolerance
        while change >= tolerance:
            change = 0.0
            # Iterate through each node (skipping the first and last)
            for i in range(1, len(path) - 1):
                # Smoothing for x coordinate
                original_x = new_path[i].x
                new_path[i].x += (
                        weight_data * (path[i].x - new_path[i].x)
                        + weight_smooth * (new_path[i - 1].x + new_path[i + 1].x - 2.0 * new_path[i].x)
                )
                change += abs(original_x - new_path[i].x)

                # Smoothing for y coordinate
                original_y = new_path[i].y
                new_path[i].y += (
                        weight_data * (path[i].y - new_path[i].y)
                        + weight_smooth * (new_path[i - 1].y + new_path[i + 1].y - 2.0 * new_path[i].y)
                )
                change += abs(original_y - new_path[i].y)

        return new_path

    def compute_path_metrics(self, path):
        def curvature(node1, node2, node3):
            a = distance(node1, node2)
            b = distance(node2, node3)
            c = distance(node1, node3)
            s = (a + b + c) / 2  # Semi-perimeter
            area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
            if area == 0:
                return 0
            return 4 * area / (a * b * c)

        # for node in path:
        # print(node)
        for i in range(1, len(path) - 1):
            path[i].curvature = curvature(path[i - 1], path[i], path[i + 1])

    def straight_path_exists(self, start, goal) -> bool:
        dx = goal.x - start.x
        dy = goal.y - start.y
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
