import numpy as np
import time
from Mark.environment import Checkpoint


def distance(point1, point2) -> float:
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


class PathExecution:
    def __init__(self, env, path_creation) -> None:
        self.env = env
        self.total_distance_to_node = 0
        self.current_next_node = None
        self.current_checkpoint_idx = 0

        self.max_lookahead_distance = 0.3
        self.min_lookahead_distance = 0.05
        self.current_lookahead_distance = self.max_lookahead_distance

        self.path = []
        self.path_creation = path_creation
        self.lookahead_point = (0, 0)

        self.current_speed = 0
        self.previous_move_time = 0

        self.counter = 0

    def update_path(self):
        if not self.path:
            return
        if distance(self.path[-1],
                    self.env.checkpoints[self.current_checkpoint_idx]) > self.env.robot.path_update_distance:
            self.path = []
            return
        for node in self.path:
            for obstacle in self.env.obstacles:
                allowed_radius = obstacle.radius + self.env.robot.radius + self.env.robot.obstacle_clearance
                if distance(node, obstacle) < allowed_radius - 0.05:  # ease up to 5 cm
                    self.path = []
                    return

    def update_lookahead_point(self):
        min_dist = float('inf')
        idx = 0
        for i in range(len(self.path)):
            dist = distance(self.path[i], self.env.robot)
            if dist < min_dist:
                min_dist = dist
                idx = i
                self.current_lookahead_distance = dist

        if distance(self.env.robot, self.env.checkpoints[self.current_checkpoint_idx]) < min_dist:
            self.current_lookahead_distance = dist
            self.lookahead_point = (self.env.checkpoints[self.current_checkpoint_idx].x,
                                    self.env.checkpoints[self.current_checkpoint_idx].y)
            return

        # Define coefficients for the cubic interpolation
        # a = 2 * (self.max_lookahead_distance - self.min_lookahead_distance) / (np.pi / 2)**3
        a = (self.min_lookahead_distance - self.max_lookahead_distance) / (np.pi / 2) ** (1 / 3)
        d = self.max_lookahead_distance

        for i in range(idx + 1, len(self.path)):
            dist = distance(self.env.robot, self.path[i])
            if dist < self.min_lookahead_distance:
                idx = i
                self.current_lookahead_distance = dist
                continue

            dx = self.path[i].x - self.env.robot.x
            dy = self.path[i].y - self.env.robot.y

            angle_difference = np.arctan2(dy, dx) - self.env.robot.a
            angle_difference = np.arctan2(np.sin(angle_difference), np.cos(angle_difference))

            # Calculate the lookahead distance using cubic interpolation
            probable_lookahead = a * abs(angle_difference) ** (1 / 3) + d

            if dist < probable_lookahead:
                idx = i
                self.current_lookahead_distance = dist
        # Set the lookahead point
        self.lookahead_point = (self.path[idx].x, self.path[idx].y)

        dist = distance(self.env.robot, self.env.checkpoints[self.current_checkpoint_idx])
        if dist < self.min_lookahead_distance:
            self.current_lookahead_distance = dist
            self.lookahead_point = (self.env.checkpoints[self.current_checkpoint_idx].x,
                                    self.env.checkpoints[self.current_checkpoint_idx].y)
            return

        dx = self.env.checkpoints[self.current_checkpoint_idx].x - self.env.robot.x
        dy = self.env.checkpoints[self.current_checkpoint_idx].y - self.env.robot.y

        angle_difference = np.arctan2(dy, dx) - self.env.robot.a
        angle_difference = np.arctan2(np.sin(angle_difference), np.cos(angle_difference))

        # Calculate the lookahead distance using cubic interpolation
        probable_lookahead = a * abs(angle_difference) ** (1 / 3) + d

        if dist < probable_lookahead:
            self.current_lookahead_distance = dist
            self.lookahead_point = (self.env.checkpoints[self.current_checkpoint_idx].x,
                                    self.env.checkpoints[self.current_checkpoint_idx].y)
            return

    def get_closest_node_curvature(self):
        min_dist = 100
        closest_node = self.path[0]
        for node in self.path:
            dist = distance(node, self.env.robot)
            if dist < min_dist:
                min_dist = dist
                closest_node = node

        return closest_node.curvature

    def get_to_desired_speed(self, speed):
        if self.previous_move_time == 0:
            self.previous_move_time = time.time()

        possible_change = (time.time() - self.previous_move_time) * self.env.robot.linear_acceleration
        if self.current_speed < speed:
            self.current_speed = min(self.current_speed + possible_change, speed)
        else:
            self.current_speed = max(self.current_speed - possible_change, speed)
        self.previous_move_time = time.time()

    def move_through_path(self):
        if distance(self.env.robot,
                    self.env.checkpoints[self.current_checkpoint_idx]) < self.env.robot.distance_allowance:
            angle_difference = self.env.robot.a - self.env.checkpoints[self.current_checkpoint_idx].a
            angle_difference = np.arctan2(np.sin(angle_difference), np.cos(angle_difference))

            # Calculate the angular speed based on how close the robot is to the desired angle
            if abs(angle_difference) > self.env.robot.angle_allowance:
                # Normalize the difference within the range of 0 to 1
                normalized_diff = abs(angle_difference) / (np.pi / 4)

                # Linear interpolation between max and min angular speed
                angular_speed = (
                                        self.env.robot.max_angular_speed * 2 - self.env.robot.min_angular_speed) * normalized_diff + self.env.robot.min_angular_speed

                # Ensure angular speed does not drop below the minimum
                angular_speed = min(angular_speed, self.env.robot.max_angular_speed * 2)
                angular_speed = max(angular_speed, self.env.robot.min_angular_speed)

                direction = -np.sign(angle_difference)
                self.get_to_desired_speed(0)
                return (self.current_speed, direction * angular_speed)

            self.current_checkpoint_idx += 1
            self.path = []
            return (0, 0)

        self.update_lookahead_point()

        a = -np.tan(self.env.robot.a)
        b = 1
        c = np.tan(self.env.robot.a) * self.env.robot.x - self.env.robot.y

        x = abs(a * self.lookahead_point[0] + b * self.lookahead_point[1] + c) / np.sqrt(a ** 2 + b ** 2)

        dx = self.lookahead_point[0] - self.env.robot.x
        dy = self.lookahead_point[1] - self.env.robot.y

        between_angle = np.arctan2(dy, dx)
        between_angle = self.env.robot.a - between_angle
        between_angle = np.arctan2(np.sin(between_angle), np.cos(between_angle))
        direction = -np.sign(between_angle)

        lookahead_curvature = (2 * x) / (self.current_lookahead_distance ** 2)

        if abs(between_angle) > np.pi / 2:
            linear_speed = 0
        else:
            linear_speed_scale = ((np.pi / 2) - abs(between_angle)) / \
                                 (np.pi / 2) * self.current_lookahead_distance / self.max_lookahead_distance
            linear_speed = self.env.robot.linear_speed * linear_speed_scale
            linear_speed = max(self.env.robot.min_linear_speed, linear_speed)

        self.get_to_desired_speed(linear_speed)

        if self.current_lookahead_distance < self.env.robot.angular_speed_distance_allowance:
            angular_speed = 0
        else:
            if abs(between_angle) < np.pi / 2:
                angular_speed = self.env.robot.max_angular_speed * lookahead_curvature
            else:
                angular_speed = self.env.robot.max_angular_speed

            angular_speed = min(angular_speed, self.env.robot.max_angular_speed)
            angular_speed = direction * angular_speed
        return self.current_speed, angular_speed

    def get_current_move(self):
        # if self.env.found_finish:
        #     self.get_to_desired_speed(0)
        #     return (self.current_speed, 0)
        if not self.path:
            # Still have checkpoints to go through
            if self.current_checkpoint_idx < len(self.env.checkpoints):
                goal = self.env.checkpoints[self.current_checkpoint_idx]
                self.path = self.path_creation.create_path(self.env.robot, goal)
                if not self.path:
                    print("Couldn't generate a path to the checkpoint")
                    self.get_to_desired_speed(0)
                    return (self.current_speed, 0)
            else:
                if not self.env.found_finish and self.counter > 10:
                    print("Adding a new checkpoint for exploration")
                    exploration_distance = 0.30
                    # self.get_to_desired_speed(0)
                    new_x = self.env.robot.x + np.cos(self.env.robot.a) * exploration_distance
                    new_y = self.env.robot.y + np.sin(self.env.robot.a) * exploration_distance
                    new_a = float(self.env.robot.a)
                    self.env.checkpoints.append(Checkpoint(new_x, new_y, new_a))
                    self.counter = 0
                    return (self.current_speed, 0)
                else:
                    self.get_to_desired_speed(0)
                    self.counter += 1
                return (self.current_speed, 0)
        # Here the path is guaranteed to be non-empty
        return self.move_through_path()