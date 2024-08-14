import time
import sys
import cv2
import numpy as np

from Graphics.visualization import RobotVisualization
from environment import Environment, Robot, Checkpoint, Obstacle
from PathManagement.path_execution import PathExecution
from PathManagement.path_creation import PathCreation
from SLAM.kalman import KalmanFilter

from Vision.rectangle import RectangleProcessor
from Vision.color import ColorSettings, ColorQueue

# Running pygame without the display to resolve a dependency with OpenGL
import pygame
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'


class TurtlebotApp:
    def __init__(self, env, visualization=False, turtlebot=False, camera_view=False):
        self.visualization = visualization
        self.turtlebot = turtlebot
        self.camera_view = camera_view

        self.env = env
        self.path_creation = PathCreation(self.env)
        self.path_execution = PathExecution(self.env, self.path_creation)
        self.kalman_filter = KalmanFilter(self.env)

        if self.visualization:
            if self.turtlebot:
                screen_dimensions = (500, 500)
                rendering_size = (500, 500)
            else:
                screen_dimensions = (800, 800)
                rendering_size = (800, 800)
            self.vis = RobotVisualization(screen_dimensions, rendering_size, self.env, self.path_execution, self.kalman_filter)

        self.max_update_rate = 1000
        self.rgb_image = True
        self.point_cloud = True
        self.depth_image = False

        self.turtle = None
        self.rate = None
        self.color_settings = None
        self.color_adapt_queue = None
        if self.turtlebot:
            self.initialize_turtlebot()

        self.next_move = (0, 0)
        self.previous_odometry = (0, 0, 0)

    def initialize_turtlebot(self):
        try:
            from robolab_turtlebot import Turtlebot, get_time, Rate
            self.turtle = Turtlebot(rgb=self.rgb_image, pc=self.point_cloud, depth=self.depth_image)
            self.rate = Rate(self.max_update_rate)

            self.initialize_robot()
            self.color_settings = ColorSettings()
            self.color_adapt_queue = ColorQueue(self.color_settings)
        except ImportError:
            print("Failed to import robolab_turtlebot. Turtlebot functionality will be disabled.")
            self.turtlebot = False

    def initialize_robot(self):
        if self.rgb_image:
            self.turtle.wait_for_rgb_image()
            print('Rgb image received')
        if self.point_cloud:
            self.turtle.wait_for_point_cloud()
            print('Point cloud received')
        if self.depth_image:
            self.turtle.wait_for_depth_image()
            print("Depth image received")

        self.turtle.wait_for_odometry()
        print("Odometry received")

        self.turtle.reset_odometry()

    def turtle_routine(self, counter):
        obstacle_measurements = []
        measurements_to_make = 0

        # if next_move[0] < 0.04 and abs(next_move[1]) < 0.01:
        #     measurements_to_make = 2
        #     print("Making two measurements")
        if abs(self.next_move[1]) < 0.01 and self.next_move[0] < 0.2:
            if self.path_execution.current_checkpoint_idx < 8:
                measurements_to_make = 5
                time.sleep(0.3)
            elif self.next_move[0] < 0.01:
                measurements_to_make = 3
                time.sleep(0.3)
            # else:
                # measurements_to_make = 1

        if measurements_to_make > 0:
            print(f"Making {measurements_to_make} measurements")
        # if next_move[0] < 0.4 and abs(next_move[1]) < 0.01:
        #     if self.path_execution.current_checkpoint_idx <= 1:
        #         print("going through first checkpoints, making 5 measurements")
        #         time.sleep(0.3)
        #     else:
        #         measurements_to_make = 1
        #         print("Making one measurement")

        for _ in range(measurements_to_make):
            image = self.turtle.get_rgb_image()
            pc_image = self.turtle.get_point_cloud()
            rectg_processor = RectangleProcessor(image, pc_image, self.color_settings, self.color_adapt_queue)
            detected_rectgs, masked_rectgs, image_rectg, _ = rectg_processor.process_image()
            if detected_rectgs is None:
                continue

            if self.camera_view:
                cv2.imshow('RGB Camera', image_rectg)
                # cv2.imshow('Combined mask', masked_rectgs)
            obstacle_measurements += detected_rectgs

        current_odometry = self.turtle.get_odometry()
        odometry_change = current_odometry - self.previous_odometry
        self.previous_odometry = current_odometry

        self.kalman_filter.pos_update(odometry_change)
        self.kalman_filter.obstacles_measurement_update(obstacle_measurements)
        self.kalman_filter.update_for_visualization()

        self.env.update_checkpoints(self.path_execution.current_checkpoint_idx)
        self.path_execution.update_path()

        self.next_move = self.path_execution.get_current_move()
        self.turtle.cmd_velocity(angular=self.next_move[1], linear=self.next_move[0])

        self.rate.sleep()


    def simulation_routine(self, dt, counter):
        next_move = self.path_execution.get_current_move()
        odometry_change = self.env.simulate_movement(next_move, dt)

        obstacle_measurements = self.env.get_measurement()

        self.kalman_filter.pos_update(odometry_change)
        self.kalman_filter.obstacles_measurement_update(obstacle_measurements)
        self.kalman_filter.update_for_visualization()

        self.env.update_checkpoints(self.path_execution.current_checkpoint_idx)
        self.path_execution.update_path()

    def run(self):
        running = True
        counter = 0
        previous_time = 0
        print("Starting main loop")
        while running:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False

            if self.visualization:
                if (counter % 2 == 0 and self.turtlebot) or not self.turtlebot:
                    self.vis.draw_everything(self.turtlebot)
                    self.vis.show_cv2(counter)
                    self.vis.clock.tick(120)

            if self.turtlebot:
                if self.turtle.is_shutting_down():
                    running = False
                self.turtle_routine(counter)
            else:
                if previous_time == 0:
                    previous_time = time.time()
                dt = time.time() - previous_time
                # if dt > 0.06:
                if dt > 0.01:
                    self.simulation_routine(dt, counter)
                    previous_time = time.time()

            counter += 1

        print("Cycles:", counter)
        cv2.destroyAllWindows()
        pygame.quit()


def main():
    # print(f"PID: {os.getpid()}")
    # input("Press Enter to continue...")
    visualization = "-vis" in sys.argv
    turtlebot = "-turtlebot" in sys.argv
    camera_view = "-camera" in sys.argv

    # Example environment for simulation testing
    if not turtlebot:
        env = Environment(Robot(0, 0, 0),
                          [
                              # Checkpoint(0, 0, angle) for angle in np.arange(0, 5 * np.pi / 2, np.pi / 2)
                          ],
                          [],
                          [
                           Obstacle(1, 0.05, 0),
                           Obstacle(1, -0.05, 1),
                           Obstacle(0.9, 0.7, 2),
                           Obstacle(1.45, 1.28, 0),
                           Obstacle(1.50, 1.25, 1),
                           Obstacle(0, 1.60, 2),
                           Obstacle(0.05, 1.65, 2)
                           ])
        # env = Environment(Robot(0, 0, 0),
        #                   [
        #                       # Checkpoint(0, 0, angle) for angle in np.arange(0, 5 * np.pi / 2, np.pi / 2)
        #                       Checkpoint(2, 0,  0),
        #                       Checkpoint(0, 0, np.pi),
        #                       Checkpoint(2, 0, 0),
        #                       Checkpoint(0, 0, np.pi),
        #                       Checkpoint(2, 0,  0),
        #                       Checkpoint(0, 0, np.pi),
        #                       Checkpoint(2, 0,  0),
        #                       Checkpoint(0, 0, np.pi),
        #                       # Checkpoint(2, 0,  0),
        #                       # Checkpoint(0, 0, np.pi),
        #                       # Checkpoint(2, 0,  0),
        #                       # Checkpoint(0, 0, np.pi),
        #                       # Checkpoint(2, 0,  0),
        #                       # Checkpoint(0, 0, np.pi),
        #                   ],
        #                   [],
        #                   [
        #                    Obstacle(1, 0.05, 0),
        #                    Obstacle(1, -0.05, 1),
        #                    Obstacle(0.50, -0.35, 2),
        #                    Obstacle(1.50, -0.35, 2),
        #                    Obstacle(0.50, 0.35, 2),
        #                    Obstacle(1.50, 0.35, 2),
        #                    ])
    else:
        env = Environment(Robot(0, 0, 0),
                          [Checkpoint(0, 0, angle) for angle in np.arange(0, 2 * np.pi + 0.01, np.pi / 4)],
                          [],
                          [])

    app = TurtlebotApp(env, visualization=visualization, turtlebot=turtlebot, camera_view=camera_view)
    app.run()


if __name__ == "__main__":
    main()
