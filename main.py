import cv2
import time
import cProfile 
import pstats
import numpy as np
import sys

from visualization import RobotVisualization
from environment import Environment, Robot, Checkpoint, Obstacle
from path_execution import PathExecution
from path_creation import PathCreation
from kalman import KalmanFilter

# Running pygame without the display to resolve a dependency with OpenGL
import pygame 
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

def initialize_turtle():
    # turtle = Turtlebot(rgb = True, depth = True, pc = True)
    
    turtle = Turtlebot()

    # turtle.wait_for_rgb_image()
    # print('Rgb image received')
    
    # turtle.wait_for_point_cloud()
    # print('Point cloud received')
    
    # turtle.wait_for_depth_image()
    # print("Depth image received")
    
    # turtle.wait_for_odometry()
    # print("Odometry received")
    
    turtle.reset_odometry()
    turtle.get_odometry()
    
    rate = Rate(1000)
    return turtle, rate

def main():

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
    #                   set(), set())   
     
    # env = Environment(Robot(0, 0, np.deg2rad(-180)),
    #                   [Checkpoint(2, 0, 0),
    #                    Checkpoint(0, 0, 0)],
    #                   [Obstacle(1, 0.2, "red"),
    #                    Obstacle(1, -0.2, "blue")], set())  
     

    env = Environment(Robot(-0.3, 0, 0), Robot(-0.3, 0, 0),
                        [Checkpoint(1, 0, 0),
                         Checkpoint(2, 1, np.pi/4),
                         Checkpoint(1, 0, -np.pi)],
                        [],
                        [Obstacle(0.50, -0.20, 0),
                         Obstacle(0.50, 0.20, 1),
                         Obstacle(0.50, -0.50, 2),
                         Obstacle(1.5, 0.50, 1),
                         Obstacle(1.5, 0.75, 2),
                         ])
    
    # env = Environment(Robot(0, 0, 0), 
    #                     [Checkpoint(1, 0, 0)],
    #                     [],
    #                     [Obstacle(0.7, 0, 0)
    #                      ])
    
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
    # env = Environment(Robot(0, 0, 0), 
    #                     [],
    #                     [],
    #                     [Obstacle(0.4, 0.1, 0)
    #                      ])
    
    path_creation = PathCreation(env)
    path_execution = PathExecution(env, path_creation)
    kalman_filter = KalmanFilter(env)
    
    if visualization:
        screen_dimensions = (2160, 3840)
        # screen_dimensions = (1440, 900)
        vis = RobotVisualization(env, path_execution, kalman_filter, screen_dimensions)
    
    if turtlebot:
        from robolab_turtlebot import Turtlebot, get_time, Rate
        turtle, rate = initialize_turtle()
    
    
    running = True
    counter = 0
    previous_time = 0
    previous_obstacles_size = 0
    previous_odometry = (0, 0, 0)
    measurements = []
    print("Starting main loop")
    # with cProfile.Profile() as pr:
    while running:   
        if visualization:       
            # if counter % 10 == 0:
            vis.screen.fill((0, 0, 0))
            vis.draw_everything()

            vis.show_cv2()
        
        if turtlebot:
            odometry = turtle.get_odometry()
            if previous_time == 0:
                previous_time = time.time()
        
            dt = time.time() - previous_time
            angular_velocity = -(previous_odometry[2] - odometry[2]) / dt
                    
            # Compute the straight-line distance between the current and previous positions
            delta_x = odometry[0] - previous_odometry[0]
            delta_y = odometry[1] - previous_odometry[1]
            distance_straight = np.sqrt(delta_x**2 + delta_y**2)

            # Calculate the linear velocity considering curvature
            # if angular_velocity > 1e-5:
            #     radius = distance_straight / angular_velocity
            #     linear_velocity = radius * angular_velocity
            # else:
            linear_velocity = distance_straight / dt
            print("Previous move velocity:", linear_velocity)
            previous_move = (linear_velocity, angular_velocity)
            previous_time = time.time()
            previous_odometry = odometry
            
            print("Odometry:", odometry)
            print("Robot position:", env.robot.x, env.robot.y, env.robot.a)

            measurements = env.get_measurement()
            if len(env.obstacles) > previous_obstacles_size:
                previous_obstacles_size = len(env.obstacles)
                print("Resetting path")
                path_execution.path = []
                previous_time = 0

            kalman_filter.process_measurement(previous_move, measurements, dt)
            env.simulate_movement(previous_move, dt)

            next_move = path_execution.get_current_move()


            turtle.cmd_velocity(angular = next_move[1], linear = next_move[0])
            rate.sleep()
            
            if (cv2.waitKey(1) & 0xFF == ord('q')) or turtle.is_shutting_down():
                running = False
        else:
            measurements = env.get_measurement()
            if len(env.obstacles) > previous_obstacles_size:
                previous_obstacles_size = len(env.obstacles)
                print("Resetting path")
                path_execution.path = []
                previous_time = 0
                
            if previous_time == 0:
                previous_time = time.time()
        
            dt = time.time() - previous_time    
            next_move = path_execution.get_current_move()
            kalman_filter.process_measurement(next_move, measurements, dt)
            env.simulate_movement(next_move, dt)
            previous_time = time.time()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False

        # rgb = turtle.get_rgb_image()
        # cv2.imshow('RGB Camera', rgb)     
         
        counter += 1
        vis.clock.tick(120)
            
    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # # stats.print_stats()
    # stats.dump_stats(filename='profiling.prof')
                
    print("Cycles:", counter)
    cv2.destroyAllWindows()
    pygame.quit()
    return        

if __name__ == "__main__":
    main()    