import pygame
import pygame.gfxdraw
import math
# from environment import Node
import numpy as np
import cv2


class RobotVisualization:
    def __init__(self, env, screen_size, window_percentage = 0.8, margin = 30):
        self.env = env
        self.margin = margin
        
        minimal_dimension = int(min(screen_size)*window_percentage)
        self.screen_size = (minimal_dimension, minimal_dimension)
        # self.screen_size = (screen_size[0]*window_percentage, screen_size[1]*window_percentage)
        
        pygame.init()
        # self.base_size = (3840, 3840) # 4K square
        self.base_size = (1080,1080)
        self.screen = pygame.display.set_mode(self.base_size)
        self.clock = pygame.time.Clock()

        self.scale_width = self.screen_size[0] / self.base_size[0]
        self.scale_height = self.screen_size[1] / self.base_size[1]

        self.window_name = 'Pygame Simulation'
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        x_y_offset = int((min(screen_size) - minimal_dimension) / 4)
        cv2.moveWindow(self.window_name, x_y_offset, x_y_offset - 20)
        
        self.max_coordinate = 0
        self.min_coordinate = 0
        self.find_limits()
        
        self.grid_surface = None
        self.initialize_grid()

    def show_cv2(self):
        pygame.display.update()
        cv2.imshow(self.window_name, self.get_cv2_image())        
        
    def get_cv2_image(self):
        raw_str = pygame.image.tostring(self.screen, 'RGB')
        image = np.frombuffer(raw_str, dtype=np.uint8).reshape(self.base_size[1], self.base_size[0], 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, None, fx = self.scale_width,
                           fy = self.scale_height, interpolation=cv2.INTER_AREA)
        return image

    def create_button(self, text, position, action):
        """Create a button in the Pygame window."""
        font = pygame.font.Font(None, 36)
        text_render = font.render(text, True, (0, 0, 0))
        text_rect = text_render.get_rect(center=position)
        button_rect = text_render.get_rect()  # Create a rect for button
        button_rect.center = position
        pygame.draw.rect(self.screen, (200, 200, 200), button_rect)  # Draw the button
        self.screen.blit(text_render, text_rect)

        return button_rect, action

    def draw_everything(self):
        # self.draw_grid()
        if self.grid_surface:
            self.screen.blit(self.grid_surface, (0, 0))

        for obstacle in self.env.obstacles:
            self.draw_obstacle(obstacle)

        for checkpoint in self.env.checkpoints:
            self.draw_checkpoint(checkpoint)

        self.draw_robot(self.env.robot)
        
    def draw_path(self, path):
        updated_rects = []  # List to store rectangles that need to be updated
        if not path:
            print("No path to draw")
            return updated_rects  # Return an empty list

        path_color = (0, 100, 200)  # Dark blue color for the path
        line_width = 5

        # Insert the robot's current position as the first node in the path
        full_path = [self.env.robot] + path

        # Draw lines between nodes and store the updated areas
        for i in range(len(full_path) - 1):
            start_node = full_path[i]
            end_node = full_path[i + 1]
            start_pos = self.get_coordinates_in_pixels(start_node.x, start_node.y)
            end_pos = self.get_coordinates_in_pixels(end_node.x, end_node.y)
            pygame.draw.line(self.screen, path_color, start_pos, end_pos, line_width)

            # Calculate the smallest rectangle that covers the line
            line_rect = pygame.Rect(min(start_pos[0], end_pos[0]), min(start_pos[1], end_pos[1]),
                                    abs(start_pos[0] - end_pos[0]), abs(start_pos[1] - end_pos[1]))
            line_rect.inflate_ip(line_width, line_width)  # Inflate to cover the whole line width
            updated_rects.append(line_rect)

        # Draw nodes and store the updated areas
        node_radius_pixels = self.get_length_in_pixels(1)
        for node in full_path:
            node_pos = self.get_coordinates_in_pixels(node.x, node.y)
            pygame.gfxdraw.aacircle(self.screen, node_pos[0], node_pos[1], node_radius_pixels, path_color)
            pygame.gfxdraw.filled_circle(self.screen, node_pos[0], node_pos[1], node_radius_pixels, path_color)

            # Calculate the rectangle that covers the node
            node_rect = pygame.Rect(node_pos[0] - node_radius_pixels, node_pos[1] - node_radius_pixels,
                                    2 * node_radius_pixels, 2 * node_radius_pixels)
            updated_rects.append(node_rect)

        return updated_rects  # Return the list of updated rectangles

    def draw_robot(self, robot):
        self.draw_arrow(robot, (50, 220, 50))
        pygame.gfxdraw.aacircle(self.screen,
                                *self.get_coordinates_in_pixels(robot.x, robot.y),
                                self.get_length_in_pixels(robot.radius),
                                (0, 255, 0))

    def draw_obstacle(self, obstacle):
        pygame.gfxdraw.aacircle(self.screen,
                                *self.get_coordinates_in_pixels(obstacle.x, obstacle.y),
                                self.get_length_in_pixels(obstacle.radius),
                                (255, 0, 0))
        pygame.gfxdraw.filled_circle(self.screen,
                                     *self.get_coordinates_in_pixels(obstacle.x, obstacle.y),
                                     self.get_length_in_pixels(obstacle.radius), 
                                     (255, 0, 0))

    def draw_checkpoint(self, checkpoint):
        self.draw_arrow(checkpoint, (220, 50, 50))
        pygame.gfxdraw.aacircle(self.screen,
                                *self.get_coordinates_in_pixels(checkpoint.x, checkpoint.y),
                                self.get_length_in_pixels(1.5), (0, 0, 255))
        pygame.gfxdraw.filled_circle(self.screen,
                                     *self.get_coordinates_in_pixels(checkpoint.x, checkpoint.y),
                                     self.get_length_in_pixels(1.5), (0, 0, 255))
    def draw_arrow(self, node, color):
        arrow_size = self.get_length_in_pixels(10)  # cm
        
        x, y = self.get_coordinates_in_pixels(node.x, node.y)
        end_line_x = x + arrow_size * 0.9 * math.cos(math.radians(node.a))  # Shorten the line slightly
        end_line_y = y - arrow_size * 0.9 * math.sin(math.radians(node.a))

        end_arrow_x = x + arrow_size * math.cos(math.radians(node.a))
        end_arrow_y = y - arrow_size * math.sin(math.radians(node.a))
        pygame.draw.line(self.screen, color, (x, y), (int(end_line_x), int(end_line_y)), self.get_length_in_pixels(0.5))
        
        # Draw arrowhead
        arrowhead_size = arrow_size // 4  # Increase arrowhead size for better visibility
        left_dx = arrowhead_size * math.cos(math.radians(node.a - 150))  # Sharper angle for the arrowhead
        left_dy = arrowhead_size * math.sin(math.radians(node.a - 150))
        right_dx = arrowhead_size * math.cos(math.radians(node.a + 150))  # Sharper angle
        right_dy = arrowhead_size * math.sin(math.radians(node.a + 150))
        
        left_end_x = end_arrow_x + left_dx
        left_end_y = end_arrow_y - left_dy
        right_end_x = end_arrow_x + right_dx
        right_end_y = end_arrow_y - right_dy

        # Draw the arrowhead as a triangle
        pygame.draw.polygon(self.screen, color, [(int(end_arrow_x), int(end_arrow_y)), 
                                                (int(left_end_x), int(left_end_y)), 
                                                (int(right_end_x), int(right_end_y))])

    def get_coordinates_in_pixels(self, x, y):
        result_x = int((self.base_size[0]*(x - self.min_coordinate))
                       /(self.max_coordinate-self.min_coordinate))
        result_y = int((self.base_size[1]*(self.max_coordinate - y))
                       /(self.max_coordinate-self.min_coordinate))
        return result_x, result_y

    def get_length_in_pixels(self, length):
        return int(self.base_size[0]*(length / abs(self.max_coordinate - self.min_coordinate)))


    def find_limits(self):
        # Collects all x and y coordinates from robots, checkpoints, and obstacles
        all_x = [self.env.robot.x] + \
        [cp.x for cp in self.env.checkpoints] + \
        [ob.x for ob in self.env.obstacles]

        all_y = [self.env.robot.y] + \
        [cp.y for cp in self.env.checkpoints] + \
        [ob.y for ob in self.env.obstacles]

        min_x, max_x = min(all_x) - self.margin, max(all_x) + self.margin
        min_y, max_y = min(all_y) - self.margin, max(all_y) + self.margin

        # self.min_x = self.min_y = min(self.min_x, self.min_y)
        # self.max_x = self.max_y = max(self.max_x, self.max_y)
        self.min_coordinate = min(min_x, min_y)
        self.max_coordinate = max(max_x, max_y)
        # print(self.max_coordinate, self.min_coordinate)

    def initialize_grid(self):
        self.grid_surface = pygame.Surface(self.screen.get_size())
        # self.grid_surface.fill((255, 255, 255))  # Fill background, adjust color as needed

        grid_color = (200, 200, 200)  # Light grey
        spacing = 25  # cm for now
        font = pygame.font.Font(None, int(self.screen_size[0]*0.04))

        # Draw vertical lines
        x = 0
        while x < self.max_coordinate:
            self.draw_grid_line_with_number(x, self.min_coordinate,
                                            x, self.max_coordinate,
                                            grid_color, font, True,
                                            self.grid_surface)
            x += spacing

        x = -spacing
        while x > self.min_coordinate:
            self.draw_grid_line_with_number(x, self.min_coordinate,
                                            x, self.max_coordinate,
                                            grid_color, font, True,
                                            self.grid_surface)
            x -= spacing

        # Draw horizontal lines
        y = 0
        while y < self.max_coordinate:
            self.draw_grid_line_with_number(self.min_coordinate, y,
                                            self.max_coordinate, y,
                                            grid_color, font, False,
                                            self.grid_surface)
            y += spacing

        y = -spacing
        while y > self.min_coordinate:
            self.draw_grid_line_with_number(self.min_coordinate, y,
                                            self.max_coordinate, y,
                                            grid_color, font, False,
                                            self.grid_surface)
            y -= spacing


    def draw_grid_line_with_number(self, x1, y1, x2, y2, color, font, vertical, surface):
        # Convert coordinates
        start_pos = self.get_coordinates_in_pixels(x1, y1)
        end_pos = self.get_coordinates_in_pixels(x2, y2)
        
        # Draw the line
        pygame.draw.line(surface, color, start_pos, end_pos)

        # Prepare the number text
        number_text = f"{x1 if vertical else y1}"
        text_surface = font.render(number_text, True, color)

        # Determine text position
        if vertical:
            text_x = start_pos[0] - text_surface.get_width() / 2
            text_y = self.get_coordinates_in_pixels(0, 0)[1] + 20 # TODO: Optimize
            text_x = max(text_x, 0)
        else:
            text_x = self.get_coordinates_in_pixels(0, 0)[0] - text_surface.get_width() - 20
            text_y = start_pos[1] - text_surface.get_height() / 2
            text_y = min(text_y, surface.get_height() - text_surface.get_height())

        surface.blit(text_surface, (text_x, text_y))