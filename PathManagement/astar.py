from typing import Optional
import numpy as np
import heapq


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.x == other.x) and (self.y == other.y)

    def __hash__(self):
        return hash((self.x, self.y))

    @property
    def __str__(self) -> str:
        return f"Node: ({self.x:.4f},{self.y:.4f})"

    def __repr__(self) -> str:
        return f"Node({self.x}, {self.y})"


def distance(point1: Node, point2: Node):
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


class PriorityQueue:
    def __init__(self, goal):
        self.goal = goal
        self.elements = []

    def put(self, item, priority=None):
        if priority is None:
            priority = self.heuristic(item, self.goal)
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

    def empty(self):
        return not self.elements

    @staticmethod
    def heuristic(node1, node2):
        return distance(node1, node2)


class AStar:
    def __init__(self, env):
        self.env = env

    @staticmethod
    def heuristic(node: Node, goal: Node):
        return distance(node, goal)

    def get_neighbors(self, node):
        step_size = 3
        directions = [(-step_size, 0), (step_size, 0), (0, -step_size), (0, step_size)]
        result = []

        for dx, dy in directions:
            new_node = Node(node.x + dx, node.y + dy)

            if not self.in_collision_a_star(new_node):
                result.append(new_node)
        return result

    def in_collision_a_star(self, node):
        for obstacle in self.env.obstacles:
            if self.env.obstacles_measurement_count[obstacle] < self.env.measurements_to_be_sure:
                continue
            allowed_distance = (obstacle.radius + self.env.robot.radius + self.env.robot.obstacle_clearance) * 100
            if distance(node, Node(obstacle.x * 100, obstacle.y * 100)) < allowed_distance:
                return True
        return False

    @staticmethod
    def reconstruct_path(came_from, start, goal):
        current = goal
        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return [Node(x=node.x / 100, y=node.y / 100) for node in path]

    def search(self, actual_goal):
        start = Node(int(self.env.robot.x * 100), int(self.env.robot.y * 100))
        goal = Node(int(actual_goal.x * 100), int(actual_goal.y * 100))
        frontier = PriorityQueue(goal)

        frontier.put(start, 0)
        came_from: dict[Node, Optional[Node]] = {start: None}
        cost_so_far: dict[Node, float] = {start: 0}

        while not frontier.empty():
            current = frontier.get()

            if distance(current, goal) <= 10:
                path = self.reconstruct_path(came_from, start, current)
                return path

            for neighbor in self.get_neighbors(current):
                new_cost = cost_so_far[current] + distance(current, neighbor)
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal)
                    frontier.put(neighbor, priority)
                    came_from[neighbor] = current

        return []
