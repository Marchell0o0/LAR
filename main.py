import time
import sys
import numpy as np
import cv2
import os

from Mark.visualization import RobotVisualization
from Mark.environment import Environment, Robot, Checkpoint, Obstacle
from Mark.path_execution import PathExecution
from Mark.path_creation import PathCreation
from Mark.kalman import KalmanFilter

from Uleh.rectangle import RectangleProcessor
from Uleh.color import ColorSettings, ColorQueue

# Running pygame without the display to resolve a dependency with OpenGL
import pygame
import os

os.environ['SDL_VIDEODRIVER'] = 'dummy'


def initialize_turtle():
    from robolab_turtlebot import Turtlebot, get_time, Rate

    turtle = Turtlebot(rgb=True, pc=True, depth=False)

    turtle.wait_for_rgb_image()
    print('Rgb image received')

    turtle.wait_for_point_cloud()
    print('Point cloud received')

    # turtle.wait_for_depth_image()
    # print("Depth image received")

    turtle.wait_for_odometry()
    print("Odometry received")

    turtle.reset_odometry()
    turtle.get_odometry()

    rate = Rate(1000)
    return turtle, rate


def main():
    # print("Running main.py")
    # print(f"PID: {os.getpid()}")
    # input("Press Enter to continue...")

    visualization = False
    turtlebot = False
    for argument in sys.argv:
        if argument == "-vis":
            visualization = True
        elif argument == "-turtlebot":
            turtlebot = True

    # env = Environment(Robot(0, 0, 0),
    #                     [Checkpoint(50, 50, 0),
    #                      Checkpoint(100, 0, 180),
    #                      Checkpoint(0, 0, 90)],
    #                     set())

    # env = Environment(Robot(0, 0, 0), Robot(0, 0, 0),
    #                   [Checkpoint(2, 0, 0)],
    #                   [], set())   

    # env = Environment(Robot(0, 0, np.deg2rad(-180)),
    #                   [Checkpoint(2, 0, 0),
    #                    Checkpoint(0, 0, 0)],
    #                   [Obstacle(1, 0.2, "red"),
    #                    Obstacle(1, -0.2, "blue")], set())  

    # env = Environment(Robot(-0.3, 0, 0), Robot(-0.3, 0, 0),
    #                     [Checkpoint(1, 0, 0),
    #                      Checkpoint(2, 1, np.pi/4),
    #                      Checkpoint(1, 0, -np.pi)],
    #                      [],
    #                     [Obstacle(0.50, -0.15, 0),
    #                      Obstacle(0.50, 0.15, 1),
    #                      Obstacle(0.50, -0.30, 2),
    #                      Obstacle(1.5, 0.50, 1),
    #                      Obstacle(1.5, 0.75, 2),
    #                      ])

    # env = Environment(Robot(0, 0, 0), Robot(0, 0, 0),
    #                     [],
    #                     [],
    #                     [Obstacle(1, 0.05, 0),
    #                      Obstacle(1, -0.05, 1)])

    # env = Environment(Robot(0, 0, np.pi/2), 
    #                     [Checkpoint(0, 1, np.pi/2)],
    #                     [],
    #                     [Obstacle(0, 0.7, 0)
    #                      ])
    # env = Environment(Robot(0, 0, -3*np.pi/4), 
    #                     [Checkpoint(-1, -1, -3*np.pi/4)],
    #                     [],
    #                     [Obstacle(-0.7, -0.7, 0)
    #                      ])
    # env = Environment(Robot(0, 0, 0), Robot(0, 0, 0),
    #                     [Checkpoint(1, 1, 0),
    #                      Checkpoint(0, 0, 0)],
    #                     [],
    #                     [Obstacle(0.5, 0.5, 0)])

    env = Environment(Robot(0, 0, 0), Robot(0, 0, 0),
                      [
                        #   Checkpoint(0, 0, np.pi / 2),
                        #   Checkpoint(0, 0, np.pi),
                        #   Checkpoint(0, 0, -np.pi / 2),
                        #   Checkpoint(0, 0, 0),
                        #   Checkpoint(0, 0, np.pi / 2),
                        #   Checkpoint(0, 0, np.pi),
                        #   Checkpoint(0, 0, -np.pi / 2),
                        #   Checkpoint(0, 0, 0)
                      ],
                      [],
                      [
                    #    Obstacle(1, 0.05, 0),
                    #    Obstacle(1, -0.05, 1),
                    #    Obstacle(0.5, 0.6, 2),
                    #    Obstacle(1.45, 1.28, 0),
                    #    Obstacle(1.50, 1.25, 1),
                    #    Obstacle(0, 1.60, 2),
                    #    Obstacle(0.05, 1.65, 2)
                       ])

    # env = Environment(Robot(0, 0, 0), Robot(0, 0, 0),
    #                   [
    #                       # Checkpoint(0, 0, np.pi/2),
    #                       #  Checkpoint(0, 0, np.pi),
    #                       #  Checkpoint(0, 0, -np.pi/2),
    #                       #  Checkpoint(0, 0, 0)
    #                   ],
    #                   [],
    #                   [])

    path_creation = PathCreation(env)
    path_execution = PathExecution(env, path_creation)
    kalman_filter = KalmanFilter(env)

    if visualization:
        # screen_dimensions = (2160, 3840)
        # screen_dimensions = (1440, 900)
        screen_dimensions = (500, 500)
        vis = RobotVisualization(env, path_execution, kalman_filter, screen_dimensions)

    if turtlebot:
        turtle, rate = initialize_turtle()

        color_settings = ColorSettings()
        color_adapt_queue = ColorQueue(color_settings)

        previous_odometry = (0, 0, 0)

    running: bool = True
    counter: int = 0
    previous_time = 0
    next_move = (0, 0)
    # counter_since_new_checkpoint: int = 0
    print("Starting main loop")
    while running:
        if visualization:
            if counter % 5 == 0:
                vis.draw_everything()
                vis.show_cv2()

        if turtlebot:
            if (cv2.waitKey(1) & 0xFF == ord('q')) or turtle.is_shutting_down():
                running = False
            # print("Deciding next move")
            next_move = path_execution.get_current_move()
            turtle.cmd_velocity(angular=next_move[1], linear=next_move[0])

            # rate.sleep()

            # robot isn't moving quickly
            # print("Getting camera readings")
            obstacle_measurements = []
            measurements_to_make = 0
            # This worked well
            # print(next_move)
            # if next_move[0] < 0.1 and abs(next_move[1]) < 0.0001:
                # measurements_to_make = 1
            if next_move[0] < 0.04 and abs(next_move[1]) < 0.01:
                measurements_to_make = 2
                print("Making three measurements")
            elif next_move[0] < 0.1 and abs(next_move[1]) < 0.01:
                measurements_to_make = 1
                print("Making one measurement")
            # elif next_move[0] < 0.03 and abs(next_move[1]) < 0.001:
                # measurements_to_make = 3
                # print("Making three measurements")
            for _ in range(measurements_to_make):
                # print("Making a measurement")
                image = turtle.get_rgb_image()
                pc_image = turtle.get_point_cloud()

                # print("Obstacle recognition")
                rectg_processor = RectangleProcessor(image,
                                                        pc_image,
                                                        color_settings,
                                                        color_adapt_queue)
                detected_rectgs, masked_rectgs, image_rectg, _ = rectg_processor.process_image()
                if detected_rectgs is None:
                    continue
                obstacle_measurements = obstacle_measurements + detected_rectgs
                # print(obstacle_measurements)
                # if image_rectg is not None:
                    # cv2.imshow('RGB Camera', image_rectg)


            # print("Measured data:", obstacle_measurements)

          
            current_odometry = turtle.get_odometry()
            odometry_change = current_odometry - previous_odometry

            # TODO: TEST THIS
            # odometry_change[2] *= 1.0989

            # start recording odometry changes right after using it
            previous_odometry = current_odometry

            # print("Kalman filter updates")
            # update mu_t-1 to mu_t
            kalman_filter.pos_update(odometry_change)

            # measurements are from mu_t
            kalman_filter.obstacles_measurement_update(obstacle_measurements)
            kalman_filter.update_for_visualization()

            env.update_checkpoints(path_execution.current_checkpoint_idx)

            path_execution.update_path()

        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
            if counter > 20:
                # print(env.robot.a)
                # print("Current path:", path_execution.path)
                next_move = path_execution.get_current_move()
                # move change, record time from here
                if previous_time == 0:
                    previous_time = time.time()
                dt = time.time() - previous_time

                # if next_move[0] == 0 and next_move[1] == 0:
                # print("Robot has stopped, making a measurement")
                # time.sleep(0.5)

                odometry_change = env.simulate_movement(next_move, dt)

                if visualization:
                    vis.clock.tick(120)

                if counter % 10 == 0:
                    obstacle_measurements = env.get_measurement()
                else:
                    obstacle_measurements = []

                # update mu_t-1 to mu_t
                kalman_filter.pos_update(odometry_change)

                # measurements are from mu_t
                kalman_filter.obstacles_measurement_update(obstacle_measurements)
                kalman_filter.update_for_visualization()

                env.update_checkpoints(path_execution.current_checkpoint_idx)

                path_execution.update_path()
                previous_time = time.time()

        counter += 1

    print("Cycles:", counter)
    cv2.destroyAllWindows()
    pygame.quit()
    return


if __name__ == "__main__":
    main()
