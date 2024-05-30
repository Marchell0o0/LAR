import pygame
import pygame.gfxdraw
import numpy as np
import cv2


class RobotVisualization:
    def __init__(self, screen_size, base_size, env, path_execution, kalman_filter, window_percentage=1, margin=0.2):
        self.env = env
        self.path_execution = path_execution
        self.kalman_filter = kalman_filter
        self.margin = margin

        pygame.init()
        self.base_size = base_size
        self.screen = pygame.display.set_mode(self.base_size)
        self.clock = pygame.time.Clock()

        minimal_dimension = int(min(screen_size) * window_percentage)
        self.screen_size = (minimal_dimension, minimal_dimension)

        self.scale_width = self.screen_size[0] / self.base_size[0]
        self.scale_height = self.screen_size[1] / self.base_size[1]

        self.window_name = 'Pygame Simulation'
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        x_y_offset = int((min(screen_size) - minimal_dimension) / 4)
        cv2.moveWindow(self.window_name, x_y_offset, x_y_offset - 20)

        self.center_x = 0
        self.center_y = 0
        self.range = 0.1
        self.range_change_threshold = 0.05
        self.center_change_threshold = 0.05
        self.find_limits()

        self.grid_surface = None
        self.initialize_grid()

    def show_cv2(self, counter):
        pygame.display.update()
        image = self.get_cv2_image()
        cv2.imshow(self.window_name, image)
        # cv2.imwrite(f"video/{counter}-shot.png", image)

    def get_cv2_image(self):
        raw_str = pygame.image.tostring(self.screen, 'RGB')
        image = np.frombuffer(raw_str, dtype=np.uint8).reshape(self.base_size[1], self.base_size[0], 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if image.shape[:2] != self.screen_size:
            print("Resizing image")
            image = cv2.resize(image, None, fx=self.scale_width,
                            fy=self.scale_height, interpolation=cv2.INTER_AREA)
        return image

    def draw_everything(self, turtlebot=False):
        self.find_limits()

        if turtlebot:
            self.screen.fill((255, 255, 255))
        else:
            self.initialize_grid()
            if self.grid_surface:
                self.screen.blit(self.grid_surface, (0, 0))

        for hidden_obstacle in self.env.hidden_obstacles:
            self.draw_obstacle(hidden_obstacle, 0, True)

        for idx, obstacle in enumerate(self.env.obstacles):
            if self.env.obstacles_measurement_count[obstacle] >= self.env.measurements_to_be_sure:
                self.draw_obstacle(obstacle, idx, False)
            else:
                self.draw_obstacle(obstacle, idx, True)

        self.draw_path(self.path_execution.path)

        for checkpoint in self.env.checkpoints:
            self.draw_checkpoint(checkpoint)

        self.draw_robot(self.env.robot)
        if not turtlebot:
            self.draw_robot(self.env.real_robot, True)

    def draw_path(self, path):
        updated_rects = []  # List to store rectangles that need to be updated
        if not path:
            return updated_rects  # Return an empty list

        path_color = (0, 100, 200)  # Dark blue color for the path
        line_width = max(1, int(self.get_length_in_pixels(0.01)))  # Ensure line width is at least 1

        node_radius_pixels = max(1, int(self.get_length_in_pixels(0.01)))  # Ensure node radius is at least 1

        # Draw lines between nodes and store the updated areas
        for i in range(len(path) - 1):
            start_node = path[i]
            end_node = path[i + 1]
            start_pos = self.get_coordinates_in_pixels(start_node.x, start_node.y)
            end_pos = self.get_coordinates_in_pixels(end_node.x, end_node.y)
            pygame.draw.line(self.screen, path_color, start_pos, end_pos, line_width)

            # Calculate the smallest rectangle that covers the line
            line_rect = pygame.Rect(
                min(start_pos[0], end_pos[0]), min(start_pos[1], end_pos[1]),
                abs(start_pos[0] - end_pos[0]), abs(start_pos[1] - end_pos[1])
            )
            line_rect.inflate_ip(line_width, line_width)  # Inflate to cover the whole line width
            updated_rects.append(line_rect)

        # Draw nodes and store the updated areas
        for node in path:
            node_pos = self.get_coordinates_in_pixels(node.x, node.y)
            pygame.draw.circle(self.screen, path_color, node_pos, node_radius_pixels)

            # Calculate the rectangle that covers the node
            node_rect = pygame.Rect(
                node_pos[0] - node_radius_pixels, node_pos[1] - node_radius_pixels,
                2 * node_radius_pixels, 2 * node_radius_pixels
            )
            updated_rects.append(node_rect)

        lookahead_pos = self.get_coordinates_in_pixels(self.path_execution.lookahead_point[0],
                                                       self.path_execution.lookahead_point[1])
        pygame.draw.circle(self.screen, (255, 0, 0), lookahead_pos, node_radius_pixels)

        # Calculate the rectangle that covers the lookahead point
        lookahead_rect = pygame.Rect(
            lookahead_pos[0] - node_radius_pixels, lookahead_pos[1] - node_radius_pixels,
            2 * node_radius_pixels, 2 * node_radius_pixels
        )
        updated_rects.append(lookahead_rect)

        # Draw line between robot and lookahead point
        robot_pos = self.get_coordinates_in_pixels(self.env.robot.x, self.env.robot.y)
        pygame.draw.line(self.screen, (255, 0, 0), robot_pos, lookahead_pos, line_width)

        # Calculate the smallest rectangle that covers the line
        line_rect = pygame.Rect(
            min(robot_pos[0], lookahead_pos[0]), min(robot_pos[1], lookahead_pos[1]),
            abs(robot_pos[0] - lookahead_pos[0]), abs(robot_pos[1] - lookahead_pos[1])
        )
        line_rect.inflate_ip(line_width, line_width)  # Inflate to cover the whole line width
        updated_rects.append(line_rect)

        return updated_rects  # Return the list of updated rectangles

    def draw_ellipse_from_covariance(self, center, covariance, n_std):
        color = (255, 0, 0)
        pos_covariance = covariance[:2, :2]
        eigvals, eigvecs = np.linalg.eig(pos_covariance)

        axis_lengths = 2 * n_std * np.sqrt(eigvals)
        width, height = axis_lengths[0], axis_lengths[1]

        # Convert eigenvalues to pixels assuming get_length_in_pixels converts correctly
        width_pixels, height_pixels = self.get_length_in_pixels(width) / 10, self.get_length_in_pixels(height) / 10

        # Angle to rotate ellipse (convert eigenvector angle to degrees)
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        angle_degrees = np.degrees(angle)

        # Create a new surface to draw the unrotated ellipse
        ellipse_surface = pygame.Surface((int(width_pixels * 2), int(height_pixels * 2)), pygame.SRCALPHA)

        pygame.draw.ellipse(ellipse_surface, color, (0, 0, width_pixels * 2, height_pixels * 2), 1)

        # Rotate the surface
        rotated_surface = pygame.transform.rotate(ellipse_surface,
                                                  -angle_degrees)  # Negative for correct rotation direction
        new_rect = rotated_surface.get_rect(center=center)  # Get new rect for blitting

        # Blit the rotated surface onto the main screen
        self.screen.blit(rotated_surface, new_rect)

    def draw_robot(self, robot, real=False):
        # Existing code to draw the robot
        if real:
            color = (0, 70, 0)
        else:
            color = (100, 220, 100)
        self.draw_arrow(robot, color)
        robot_center = self.get_coordinates_in_pixels(robot.x, robot.y)
        robot_radius_pixels = self.get_length_in_pixels(robot.radius)
        pygame.gfxdraw.aacircle(self.screen, robot_center[0], robot_center[1], robot_radius_pixels, color)
        if not real:
            self.draw_ellipse_from_covariance(robot_center,
                                              self.kalman_filter.sigma,
                                              n_std=2)

        # Calculate FOV visualization
        fov_radius_pixels = self.get_length_in_pixels(robot.max_detection_range)
        half_fov = robot.fov_angle / 2
        # Calculate start and end angles in radians
        start_angle_rad = robot.a - half_fov
        end_angle_rad = robot.a + half_fov

        # Calculate points on the circumference of the FOV arc
        start_point = (
            int(robot_center[0] + fov_radius_pixels * np.cos(start_angle_rad)),
            int(robot_center[1] - fov_radius_pixels * np.sin(start_angle_rad))
        )
        end_point = (
            int(robot_center[0] + fov_radius_pixels * np.cos(end_angle_rad)),
            int(robot_center[1] - fov_radius_pixels * np.sin(end_angle_rad))
        )

        # Draw the FOV arc
        rect = pygame.Rect(robot_center[0] - fov_radius_pixels, robot_center[1] - fov_radius_pixels,
                           2 * fov_radius_pixels, 2 * fov_radius_pixels)
        pygame.draw.arc(self.screen, color, rect, start_angle_rad, end_angle_rad, 1)

        # Optional: Draw lines to denote the edges of the FOV
        pygame.draw.line(self.screen, color, robot_center, start_point, 1)
        pygame.draw.line(self.screen, color, robot_center, end_point, 1)

        # # Draw lookahead distance function
        # d_max = self.path_execution.max_lookahead_distance
        # d_min = self.path_execution.min_lookahead_distance
        # p = 1 / 10
        #
        # phi = np.linspace(-np.pi, np.pi, 200)
        # d = np.piecewise(phi,
        #                  [np.abs(phi) < 0.02,
        #                   (np.abs(phi) >= 0.02) & (np.abs(phi) <= np.pi / 2),
        #                   np.abs(phi) > np.pi / 2],
        #                  [d_max,
        #                   lambda phi: (d_min - d_max) / (np.pi / 2) ** p * (np.abs(phi) - 0.02) ** p + d_max,
        #                   d_min])
        #
        # # Adjust phi by the robot's orientation angle to rotate the function
        # phi += robot.a
        #
        # # Convert polar to Cartesian coordinates
        # x = d * np.cos(phi)
        # y = d * np.sin(phi)
        #
        # # Use get_coordinates_in_pixels to accurately place the points
        # lookahead_points = [self.get_coordinates_in_pixels(robot.x + x[i], robot.y + y[i]) for i in range(len(x))]
        #
        # # Draw the lookahead distance function
        # for i in range(len(lookahead_points) - 1):
        #     pygame.draw.line(self.screen, (0, 0, 255), lookahead_points[i], lookahead_points[i + 1], 1)

    def draw_obstacle(self, obstacle, index, hidden):
        color = [255, 255, 255]

        if obstacle.color == 0:
            color = [255, 0, 0]
        elif obstacle.color == 1:
            color = [0, 0, 255]
        elif obstacle.color == 2:
            color = [0, 255, 0]

        if hidden:
            color = [max(0, c - 155) for c in color]

        color = tuple(color)
        obstacle_pos = self.get_coordinates_in_pixels(obstacle.x, obstacle.y)
        obstacle_radius_pixels = self.get_length_in_pixels(obstacle.radius)
        pygame.draw.circle(self.screen, color, obstacle_pos, obstacle_radius_pixels)

        allowed_path_radius = self.get_length_in_pixels(obstacle.radius + self.env.robot.radius + self.env.robot.obstacle_clearance)
        pygame.draw.circle(self.screen, color, obstacle_pos, allowed_path_radius, 1)

        if not hidden:
            idx = 3 + 3 * index
            covariance = self.kalman_filter.sigma[idx:idx + 3, idx:idx + 3]
            self.draw_ellipse_from_covariance(obstacle_pos, covariance, 2)

    def draw_checkpoint(self, checkpoint):
        self.draw_arrow(checkpoint, (220, 50, 50))
        checkpoint_pos = self.get_coordinates_in_pixels(checkpoint.x, checkpoint.y)
        checkpoint_radius = self.get_length_in_pixels(0.015)
        pygame.draw.circle(self.screen, (0, 0, 255), checkpoint_pos, checkpoint_radius)

    def draw_arrow(self, node, color):
        arrow_size = self.get_length_in_pixels(0.1)  # cm

        x, y = self.get_coordinates_in_pixels(node.x, node.y)
        end_line_x = x + arrow_size * 0.9 * np.cos(node.a)  # Shorten the line slightly
        end_line_y = y - arrow_size * 0.9 * np.sin(node.a)

        end_arrow_x = x + arrow_size * np.cos(node.a)
        end_arrow_y = y - arrow_size * np.sin(node.a)
        pygame.draw.line(self.screen, color, (x, y), (int(end_line_x), int(end_line_y)),
                         self.get_length_in_pixels(0.005))

        # Draw arrowhead
        arrowhead_size = arrow_size // 4  # Increase arrowhead size for better visibility
        left_dx = arrowhead_size * np.cos(node.a - np.deg2rad(150))  # Sharper angle for the arrowhead
        left_dy = arrowhead_size * np.sin(node.a - np.deg2rad(150))
        right_dx = arrowhead_size * np.cos(node.a + np.deg2rad(150))  # Sharper angle
        right_dy = arrowhead_size * np.sin(node.a + np.deg2rad(150))

        left_end_x = end_arrow_x + left_dx
        left_end_y = end_arrow_y - left_dy
        right_end_x = end_arrow_x + right_dx
        right_end_y = end_arrow_y - right_dy

        # Draw the arrowhead as a triangle
        pygame.draw.polygon(self.screen, color, [(int(end_arrow_x), int(end_arrow_y)),
                                                 (int(left_end_x), int(left_end_y)),
                                                 (int(right_end_x), int(right_end_y))])

    def get_coordinates_in_pixels(self, x, y):
        result_x = int(self.base_size[0] * (x - self.center_x + self.range / 2) / self.range)
        result_y = int(self.base_size[1] * (-y + self.center_y + self.range / 2) / self.range)
        return result_x, result_y

    def get_length_in_pixels(self, length):
        return int(self.base_size[0] * (length / self.range))

    def find_limits(self):
        # Collect all x and y coordinates from robots, checkpoints, and obstacles
        all_x = [self.env.robot.x] + \
                [ob.x for ob in self.env.obstacles] + \
                [cp.x for cp in self.env.checkpoints] + \
                [node.x for node in self.path_execution.path]

        all_y = [self.env.robot.y] + \
                [ob.y for ob in self.env.obstacles] + \
                [cp.y for cp in self.env.checkpoints] + \
                [node.y for node in self.path_execution.path]

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        # Find the largest range to ensure a square aspect ratio
        range_x = max_x - min_x
        range_y = max_y - min_y
        new_range = max(range_x, range_y) + 2 * self.margin
        # new_range = max(2, new_range)

        # Calculate new min and max for both x and y to be centered
        new_center_x = (max_x + min_x) / 2
        new_center_y = (max_y + min_y) / 2

        # Update range if the change is greater than the threshold
        if abs(new_range - self.range) > self.range_change_threshold:
            self.range = new_range

        # Update center_x if the change is greater than the threshold
        if abs(new_center_x - self.center_x) > self.center_change_threshold:
            self.center_x = new_center_x

        # Update center_y if the change is greater than the threshold
        if abs(new_center_y - self.center_y) > self.center_change_threshold:
            self.center_y = new_center_y

    def initialize_grid(self):
        self.grid_surface = pygame.Surface(self.screen.get_size())
        self.grid_surface.fill((255, 255, 255))  # Fill background, adjust color as needed

        grid_color = (150, 150, 150)  # Light grey
        spacing = 0.25  # cm for now
        max_x = self.center_x + self.range / 2
        min_x = self.center_x - self.range / 2
        max_y = self.center_y + self.range / 2
        min_y = self.center_y - self.range / 2
        font = pygame.font.Font(None, int(self.base_size[0] * 0.02))

        # Draw vertical lines
        x = min_x - (min_x % spacing)
        while x < max_x:
            self.draw_grid_line_with_number(x, min_y,
                                            x, max_y,
                                            grid_color, font, True,
                                            self.grid_surface)
            x += spacing

        # Draw horizontal lines
        y = min_y - (min_y % spacing)
        while y < max_y:
            self.draw_grid_line_with_number(min_x, y,
                                            max_x, y,
                                            grid_color, font, False,
                                            self.grid_surface)
            y += spacing

    def draw_grid_line_with_number(self, x1, y1, x2, y2, color, font, vertical, surface):
        # Convert coordinates
        start_pos = self.get_coordinates_in_pixels(x1, y1)
        end_pos = self.get_coordinates_in_pixels(x2, y2)

        # Draw the line
        pygame.draw.line(surface, color, start_pos, end_pos)

        # Prepare the number text
        number_text = f"{int(100 * x1) if vertical else int(100 * y1)}"
        text_surface = font.render(number_text, True, color)
        offset = int(self.base_size[0] * 0.01)

        # Determine text position
        if vertical:
            text_x = start_pos[0] - text_surface.get_width() / 2
            # Ensure the text stays within the visible area
            text_x = max(min(text_x, surface.get_width() - text_surface.get_width()), 0)
            text_y = end_pos[1] + offset if start_pos[1] > end_pos[1] else start_pos[
                                                                               1] - text_surface.get_height() - offset
        else:
            text_y = start_pos[1] - text_surface.get_height() / 2
            text_y = max(min(text_y, surface.get_height() - text_surface.get_height()), 0)
            text_x = end_pos[0] + offset if start_pos[0] > end_pos[0] else start_pos[
                                                                               0] - text_surface.get_width() - offset

        # Clamp text position to the edges if it is off-screen
        text_x = clamp(text_x, 0, surface.get_width() - text_surface.get_width())
        text_y = clamp(text_y, 0, surface.get_height() - text_surface.get_height())

        surface.blit(text_surface, (text_x, text_y))


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)
