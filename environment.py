import numpy as np


def distance(point1, point2) -> float:
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


class Robot:
    def __init__(self, x, y, a, ):
        self.radius = 0.177  # m

        self.max_detection_range = 1.5  # m
        self.fov_angle = np.deg2rad(70)

        # self.linear_acceleration = 0.4  # m/s^2
        # self.max_linear_speed = 0.4  # m/s
        # self.max_angular_speed = np.pi / 4  # rad/s
        # self.min_angular_speed = np.pi / 10
        # self.min_linear_speed = 0.03

        self.linear_acceleration = 0.5  # m/s^2
        self.max_linear_speed = 0.4  # m/s
        self.max_angular_speed = np.pi / 4 # rad/s
        self.min_angular_speed = np.pi / 6
        self.min_linear_speed = 0.03

        self.distance_allowance = 0.03
        self.path_update_distance = 0.02
        self.angle_allowance = np.deg2rad(1)
        # self.obstacle_clearance = 0.045  # m
        self.obstacle_clearance = 0.045  # m

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

    def __repr__(self) -> str:
        return f'Checkpoint({self.x, self.y, self.a})'


class Obstacle:
    def __init__(self, x, y, color, radius=0.025):
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
        self.obstacles_measurement_count: dict[Obstacle, int] = {}
        self.primary_checkpoints_idxs: list[int] = []
        self.real_robot: Robot = Robot(robot.x, robot.y, robot.a)

        self.measurements_to_be_sure = 3

        self.found_finish = False

        self.look_around_angle = np.pi / 8

        self.hidden_obstacles: set[Obstacle] = hidden_obstacles

    def generate_checkpoints_for_exploration(self, main_checkpoint: Checkpoint, side: int):
        turn_to_the_side = Checkpoint(main_checkpoint.x, main_checkpoint.y, main_checkpoint.a - side * np.pi / 2)
        turn_to_the_side.a = np.arctan2(np.sin(turn_to_the_side.a), np.cos(turn_to_the_side.a))

        move_forward = 0.3
        move_forward = Checkpoint(turn_to_the_side.x + move_forward * np.cos(turn_to_the_side.a),
                                  turn_to_the_side.y + move_forward * np.sin(turn_to_the_side.a),
                                  turn_to_the_side.a)
        look_around_right_one = Checkpoint(move_forward.x, move_forward.y, move_forward.a - self.look_around_angle)
        look_around_right_one.a = np.arctan2(np.sin(look_around_right_one.a), np.cos(look_around_right_one.a))

        look_around_left_one = Checkpoint(move_forward.x, move_forward.y, move_forward.a + self.look_around_angle)
        look_around_left_one.a = np.arctan2(np.sin(look_around_left_one.a), np.cos(look_around_left_one.a))

        return [turn_to_the_side, move_forward, look_around_right_one, look_around_left_one, move_forward]

    def add_checkpoint_for_obstacle_pair(self, obstacle_one, obstacle_two, current_checkpoint_idx):
        center = ((obstacle_one.x + obstacle_two.x) / 2, (obstacle_one.y + obstacle_two.y) / 2)
        normal = (-obstacle_two.y + obstacle_one.y, obstacle_two.x - obstacle_one.x)
        side = normal[0] * (self.robot.x - obstacle_one.x) + normal[1] * (self.robot.y - obstacle_one.y)

        finish_node = False
        if obstacle_one.color == 2 and obstacle_two.color == 2:
            finish_node = True

        if side == 0:
            side = 0.0001

        checkpoint_angle = np.arctan2(np.sign(side) * normal[1], np.sign(side) * normal[0])
        checkpoint_angle += np.pi
        checkpoint_angle = np.arctan2(np.sin(checkpoint_angle), np.cos(checkpoint_angle))

        offset_distance = 0.05 + self.robot.radius + obstacle_one.radius
        new_checkpoint = Checkpoint(center[0] - offset_distance * np.cos(checkpoint_angle),
                                    center[1] - offset_distance * np.sin(checkpoint_angle),
                                    checkpoint_angle)
        safety_checkpoint = Checkpoint(center[0] - (offset_distance + 0.15) * np.cos(checkpoint_angle),
                                    center[1] - (offset_distance + 0.15) * np.sin(checkpoint_angle),
                                    checkpoint_angle)

        if not finish_node:
            additional_checkpoints = self.generate_checkpoints_for_exploration(new_checkpoint, np.sign(side))

        found = False
        for idx in self.primary_checkpoints_idxs:
            distance_from_center = np.sqrt((center[0] - self.checkpoints[idx].x) ** 2 +
                                           (center[1] - self.checkpoints[idx].y) ** 2)
            if abs(distance_from_center - offset_distance) > 0.2:
                continue
            found = True

            if idx > current_checkpoint_idx - 4:
                self.checkpoints[idx - 1] = safety_checkpoint
                self.checkpoints[idx] = new_checkpoint

                if finish_node:
                    break

                for i, checkpoint in enumerate(additional_checkpoints):
                    self.checkpoints[idx + i + 1] = additional_checkpoints[i]
                break

        if self.found_finish and finish_node and not found:
            print("Adding new finish checkpoint")
            self.checkpoints.append(safety_checkpoint)
            self.checkpoints.append(new_checkpoint)
            self.primary_checkpoints_idxs.append(len(self.checkpoints) - 1)
            return True
        elif not found and not self.found_finish:
            print("Adding new checkpoint")
            self.checkpoints.append(safety_checkpoint)
            self.checkpoints.append(new_checkpoint)
            self.primary_checkpoints_idxs.append(len(self.checkpoints) - 1)
            if not finish_node:
                self.checkpoints = self.checkpoints + additional_checkpoints
            return True
        return False

    def update_checkpoints(self, current_checkpoint_idx):
        obstacles_red = [obstacle for obstacle in self.obstacles if
                         (obstacle.color == 0 and
                          self.obstacles_measurement_count[obstacle] >= self.measurements_to_be_sure)]
        obstacles_blue = [obstacle for obstacle in self.obstacles if
                          (obstacle.color == 1 and
                           self.obstacles_measurement_count[obstacle] >= self.measurements_to_be_sure)]
        obstacles_green = [obstacle for obstacle in self.obstacles if
                           (obstacle.color == 2 and
                            self.obstacles_measurement_count[obstacle] >= self.measurements_to_be_sure)]
        added_new = False
        for obstacle_red in obstacles_red:
            for obstacle_blue in obstacles_blue:
                if abs(distance(obstacle_blue, obstacle_red) - 0.055) < 0.2:
                    if self.add_checkpoint_for_obstacle_pair(obstacle_red,
                                                             obstacle_blue,
                                                             current_checkpoint_idx):
                        added_new = True

        for i in range(len(obstacles_green)):
            for j in range(len(obstacles_green)):
                if i == j:
                    continue
                if abs(distance(obstacles_green[i], obstacles_green[j]) - 0.055) < 0.2:
                    if not self.found_finish:
                        print("--------------- Found FINISH ---------------")
                        self.found_finish = True
                    self.add_checkpoint_for_obstacle_pair(obstacles_green[i],
                                                          obstacles_green[j],
                                                          current_checkpoint_idx)
        return added_new

    def simulate_movement(self, move, time):
        # Introduce noise in linear and angular velocities
        # linear_noise = np.random.normal(0, 0.05)  # Mean 0, standard deviation 0.05
        # angular_noise = np.random.normal(0, 0.1)  # Mean 0, standard deviation 0.01

        # Apply the noise to the movement values
        noisy_linear = move[0]
        noisy_angular = move[1]

        measured_delta_x = np.cos(self.real_robot.a) * move[0] * time
        measured_delta_y = np.sin(self.real_robot.a) * move[0] * time
        measured_delta_theta = move[1] * time

        noisy_delta_x = np.cos(self.real_robot.a) * noisy_linear * time
        noisy_delta_y = np.sin(self.real_robot.a) * noisy_linear * time
        noisy_delta_theta = noisy_angular * time
        # Update the robot's position and orientation with the noisy values
        self.real_robot.x += measured_delta_x
        self.real_robot.y += measured_delta_y
        self.real_robot.a += measured_delta_theta

        # Normalize the angle
        self.real_robot.a = np.arctan2(np.sin(self.real_robot.a), np.cos(self.real_robot.a))

        return (noisy_delta_x, noisy_delta_y, noisy_delta_theta)

    def get_measurement(self):
        measurements = []
        for obstacle in self.hidden_obstacles:
            # Calculate true distance and angle to the obstacle
            true_distance_to_obstacle = distance(self.real_robot, obstacle)
            true_angle_to_obstacle = np.arctan2(obstacle.y - self.real_robot.y,
                                                obstacle.x - self.real_robot.x) - self.real_robot.a

            # Check if the obstacle is within the detectable range and field of view
            if (true_distance_to_obstacle <= self.real_robot.max_detection_range and
                    -self.real_robot.fov_angle / 2 <= true_angle_to_obstacle <= self.real_robot.fov_angle / 2):
                # Add noise to distance and angle measurements
                distance_noise = 0 * np.random.normal(0, 0.015)
                angle_noise = 0 * np.random.normal(0, np.radians(0.1))

                # Ensure that the measurements are scalar
                measured_distance = min(self.real_robot.max_detection_range,
                                        float(true_distance_to_obstacle + distance_noise))
                measured_angle = min(self.real_robot.fov_angle, float(true_angle_to_obstacle + angle_noise))

                measurements.append([measured_distance, measured_angle, float(obstacle.color)])

        return measurements

    def get_odometry(self):
        return self.real_robot.x, self.real_robot.y, self.real_robot.a
