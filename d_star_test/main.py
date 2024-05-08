import heapq

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]

class DStarLite:
    def __init__(self, start, goal, grid):
        self.start = start
        self.goal = goal
        self.grid = grid
        self.g = {}
        self.rhs = {}
        self.U = PriorityQueue()
        self.km = 0
        self.last = start
        
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                self.g[(row, col)] = float('inf')
                self.rhs[(row, col)] = float('inf')
        self.rhs[goal] = 0
        self.U.put(goal, self.calculate_key(goal))
    
    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def calculate_key(self, s):
        g_rhs = min(self.g[s], self.rhs[s])
        return (g_rhs + self.heuristic(self.start, s) + self.km, g_rhs)
    
    def update_vertex(self, u):
        if u != self.goal:
            self.rhs[u] = min([self.g[v] + 1 for v in self.get_neighbors(u)])
        if self.g[u] != self.rhs[u]:
            self.U.put(u, self.calculate_key(u))
    
    def get_neighbors(self, s):
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (s[0] + dx, s[1] + dy)
            if 0 <= neighbor[0] < len(self.grid) and 0 <= neighbor[1] < len(self.grid[0]) and self.grid[neighbor[0]][neighbor[1]] == 0:
                neighbors.append(neighbor)
        return neighbors
    
    def compute_shortest_path(self):
        while not self.U.empty():
            k_old = self.U.elements[0][0]
            s = self.U.get()
            if k_old >= self.calculate_key(self.start):
                break
            
            if self.g[s] > self.rhs[s]:
                self.g[s] = self.rhs[s]
                for neighbor in self.get_neighbors(s):
                    self.update_vertex(neighbor)
            else:
                self.g[s] = float('inf')
                for neighbor in [s] + self.get_neighbors(s):
                    self.update_vertex(neighbor)
    
    def replan(self):
        self.compute_shortest_path()
        path = []
        s = self.start
        while s != self.goal:
            path.append(s)
            s = min(self.get_neighbors(s), key=lambda x: self.g[x])
        return path
    
    def update_grid(self, updated_cells):
        for cell, new_state in updated_cells.items():
            if self.grid[cell[0]][cell[1]] != new_state:
                self.grid[cell[0]][cell[1]] = new_state
                self.update_vertex(cell)
        self.km += self.heuristic(self.last, self.start)
        self.last = self.start
        self.replan()

grid = [[0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]]
start = (0, 0)
goal = (4, 4)
dstar = DStarLite(start, goal, grid)
path = dstar.replan()
print("Initial Path:", path)
updated_cells = {(2, 2): 1, (3, 2): 0}  # Example of updating grid cells
dstar.update_grid(updated_cells)
path = dstar.replan()
print("Updated Path:", path)
