import numpy as np
import time
from environment import Checkpoint


def distance(point1, point2) -> float:
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


class PathExecution:
    def __init__(self, env, path_creation) -> None:
        self.env = env
        self.total_distance_to_node = 0
        self.current_next_node = None
        self.current_checkpoint_idx = 0

        self.max_lookahead_distance = 0.4
        self.min_lookahead_distance = 0.025
        self.current_lookahead_distance = self.max_lookahead_distance

        self.path = []
        self.path_creation = path_creation
        self.lookahead_point = (0, 0)
        self.distance_to_path = 0

        self.current_speed = 0
        self.previous_move_time = 0

        self.counter = 0

    def update_path(self):
        if not self.path:
            return
        min_dist = float('inf')
        for node in self.path:
            for obstacle in self.env.obstacles:
                allowed_radius = obstacle.radius + self.env.robot.radius + self.env.robot.obstacle_clearance
                if distance(node, obstacle) < allowed_radius - 0.06:  # ease up to 5 cm
                    self.path = []
                    return
            distance_to_checkpoint = distance(node, self.env.checkpoints[self.current_checkpoint_idx])
            min_dist = min(min_dist, distance_to_checkpoint)

        if min_dist > self.env.robot.path_update_distance:
            self.path = []
            print("Checkpoint is too far away from the path")

    def update_lookahead_point(self):
        min_dist = float('inf')
        idx = 0
        for i in range(len(self.path)):
            dist = distance(self.path[i], self.env.robot)
            if dist < min_dist:
                min_dist = dist
                idx = i
                self.current_lookahead_distance = dist
                self.distance_to_path = dist
        
        # Define coefficients for the interpolation
        interpolation_power = 1 / 5
        a = ((self.min_lookahead_distance - self.max_lookahead_distance)
             / (np.pi / 2) ** interpolation_power)
        d = self.max_lookahead_distance
        counter = 0
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

            offset = 0.01
            if abs(angle_difference) < offset:
                probable_lookahead = self.max_lookahead_distance
            elif abs(angle_difference) > np.pi / 2 or min_dist > 0.1:
                probable_lookahead = self.min_lookahead_distance
            else:
                probable_lookahead = a * (abs(angle_difference)-offset) ** interpolation_power + d

            if dist < probable_lookahead:
                idx = i
                self.current_lookahead_distance = dist
                counter += 1
            else:
                if counter > 1:
                    break
        # Set the lookahead point
        self.lookahead_point = (self.path[idx].x, self.path[idx].y)

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
            self.get_to_desired_speed(0)
            # Calculate the angular speed based on how close the robot is to the desired angle
            if abs(angle_difference) > self.env.robot.angle_allowance:
                # Normalize the difference within the range of 0 to 1
                normalized_diff = abs(angle_difference) / (np.pi / 2)

                # Linear interpolation between max and min angular speed
                angular_speed = ((self.env.robot.max_angular_speed*2 - self.env.robot.min_angular_speed)
                                 * normalized_diff + self.env.robot.min_angular_speed)

                # Ensure angular speed does not drop below the minimum
                angular_speed = min(angular_speed, self.env.robot.max_angular_speed*2)
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

        if self.current_lookahead_distance < 0.0001:
            self.current_lookahead_distance = 0.0001

        lookahead_curvature = (2 * x) / (self.current_lookahead_distance ** 2)

        if abs(between_angle) > np.pi /2:
            linear_speed = 0
        elif abs(between_angle) > np.pi / 8 and self.distance_to_path > 0.2:
            linear_speed = 0
        elif self.current_lookahead_distance < self.min_lookahead_distance:
            linear_speed = self.env.robot.min_linear_speed
        else:
            power = 2
            a = (self.env.robot.min_linear_speed - self.env.robot.max_linear_speed) / ((self.min_lookahead_distance - self.max_lookahead_distance) ** power)
            linear_speed = a * (self.current_lookahead_distance - self.max_lookahead_distance) ** power + self.env.robot.max_linear_speed
            linear_speed = max(self.env.robot.min_linear_speed, linear_speed)

        self.get_to_desired_speed(linear_speed)

        if abs(between_angle) < np.pi / 2:
            angular_speed = self.env.robot.max_angular_speed * lookahead_curvature
            angular_speed = min(angular_speed, self.env.robot.max_angular_speed)
            angular_speed = direction * angular_speed
        else:
            angular_speed = self.env.robot.max_angular_speed

        return self.current_speed, angular_speed

    def make_exploration_checkpoint(self, exploration_distance):
        new_x = self.env.robot.x + np.cos(self.env.robot.a) * exploration_distance
        new_y = self.env.robot.y + np.sin(self.env.robot.a) * exploration_distance
        new_a = float(self.env.robot.a)
        new_checkpoint = Checkpoint(new_x, new_y, new_a)
        return new_checkpoint
    def get_current_move(self):
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
                    return self.move_through_path()
                
            else:
                if not self.env.found_finish and self.counter > 0:
                    print("Adding a new checkpoint for exploration")
                    exploration_distance = 0.3
                    while True:
                        new_checkpoint = self.make_exploration_checkpoint(exploration_distance)
                        allowed = True
                        for obstacle in self.env.obstacles:
                            if distance(obstacle, new_checkpoint) <= self.env.robot.radius + self.env.robot.obstacle_clearance + obstacle.radius:
                                allowed = False
                                break

                        if allowed:
                            break

                        exploration_distance += 0.3

                    turn_right = Checkpoint(new_checkpoint.x, new_checkpoint.y,
                                            new_checkpoint.a - self.env.look_around_angle)
                    turn_left = Checkpoint(new_checkpoint.x, new_checkpoint.y,
                                           new_checkpoint.a + self.env.look_around_angle)
                    self.env.checkpoints.append(new_checkpoint)
                    self.env.checkpoints.append(turn_right)
                    self.env.checkpoints.append(turn_left)
                    self.env.checkpoints.append(new_checkpoint)

                    self.counter = 0
                else:
                    self.get_to_desired_speed(0)
                    self.counter += 1
                return (self.current_speed, 0)
        else:
            return self.move_through_path()
