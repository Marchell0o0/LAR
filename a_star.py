from typing import TypeVar, List, Tuple, Dict, Optional
# from environment import Node
import math
import heapq

T = TypeVar('T')

class PriorityQueue:
    def __init__(self):
        self.elements: List[Tuple[float, T]] = []

    def empty(self) -> bool:
        return not self.elements

    def put(self, item: T, priority: float):
        heapq.heappush(self.elements, (priority, item))

    def get(self) -> T:
        return heapq.heappop(self.elements)[1]

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


class A_star:
    def __init__(self, env):
        self.env = env

    def heuristic(self, node: Node, goal: Node) -> float:
        return self.distance(node, goal)

    def distance(self, point1: Node, point2: Node) -> float:
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def get_neighbors(self, node: Node) -> List[Node]:
        step_size = 1  # Step size can be set here if not global
        directions = [(-step_size, 0), (step_size, 0), (0, -step_size), (0, step_size)]
        result = []

        for dx, dy in directions:
            new_node = Node(node.x + dx, node.y + dy)

            if not self.env.in_collision(new_node):
                result.append(new_node)
        return result


    def search(self, goal) -> Tuple[Dict[Node, Optional[Node]], Dict[Node, float]]:
        frontier = PriorityQueue()
        # start = Node(self.env.robot.x, self.env.robot.y)
        start = self.env.robot
        frontier.put(start, 0)
        came_from: Dict[Node, Optional[Node]] = {start: None}
        cost_so_far: Dict[Node, float] = {start: 0}
        
        # dict with each node's neighbors
        graph: dict[Node, List[Node]] = {}

        while not frontier.empty():
            current: Node = frontier.get()
            # print(current.x, current.y)
            if self.distance(current, goal) <= 1: # cm allowence from the goal
                print("Goal reached")
                came_from[goal] = came_from.get(current)
                break

            if not current in graph:
                graph[current] = self.get_neighbors(current)

            for neighbor in graph[current]:
                new_cost = cost_so_far[current] + self.distance(current, neighbor)
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal)
                    frontier.put(neighbor, priority)
                    came_from[neighbor] = current

        return came_from, cost_so_far
