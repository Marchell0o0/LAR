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

from Uleh.rectangle import RectangleProcessor
from Uleh.color import ColorSettings

# Running pygame without the display to resolve a dependency with OpenGL
import pygame 
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

def initialize_turtle():
    from robolab_turtlebot import Turtlebot, get_time, Rate

    # turtle = Turtlebot(rgb = True, depth = True, pc = True)

    # turtle = Turtlebot()
    turtle = Turtlebot(rgb = True, pc = True)

    turtle.wait_for_rgb_image()
    print('Rgb image received')
    
    turtle.wait_for_point_cloud()
    print('Point cloud received')
    
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
    #                     [Obstacle(0.50, -0.20, 0),
    #                      Obstacle(0.50, 0.20, 1),
    #                      Obstacle(0.50, -0.50, 2),
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
    # env = Environment(Robot(0, 0, 0), 
    #                     [],
    #                     [],
    #                     [Obstacle(0.4, 0.1, 0)
    #                      ])
    
    
    env = Environment(Robot(0, 0, 0), Robot(0, 0, 0),
                        [Checkpoint(0, 0, np.pi/2),
                         Checkpoint(0, 0, np.pi),
                         Checkpoint(0, 0, -np.pi/2),
                         Checkpoint(0, 0, 0)],
                         [],
                         [Obstacle(1, 0.05, 0),
                         Obstacle(1, -0.05, 1),
                         Obstacle(1.,1, 1),
                         Obstacle(0.95, 1.05, 0),
                         Obstacle(0, 1.50, 2),
                         Obstacle(0.05, 1.55, 2),
                         Obstacle(-1.25, 0, 2),
                         Obstacle(-1.35, 0, 2)])
    
    path_creation = PathCreation(env)
    path_execution = PathExecution(env, path_creation)
    kalman_filter = KalmanFilter(env)

    if visualization:
        # screen_dimensions = (2160, 3840)
        screen_dimensions = (1440, 900)
        vis = RobotVisualization(env, path_execution, kalman_filter, screen_dimensions)
    
    if turtlebot:
        
        turtle, rate = initialize_turtle()
        color_settings = ColorSettings()
        color_settings.calibrate_color(turtle)
    
    
    running = True
    counter = 0
    previous_odometry = (env.robot.x, env.robot.y, env.robot.a)
    previous_time =  0
    obstacle_measurements = []
    next_move = (0, 0)
    print("Starting main loop")
    # with cProfile.Profile() as pr:
    while running:   
        if visualization:       
            # if counter % 10 == 0:
            # print("Current obstacles before drawing")
            # for obstacle in env.obstacles:
            # print(obstacle)
            vis.screen.fill((0, 0, 0))
            vis.draw_everything()

            vis.show_cv2()
        

        if turtlebot:
            if (cv2.waitKey(1) & 0xFF == ord('q')) or turtle.is_shutting_down():
                running = False
            
            # compute next move
            
            # execute the move
            
            # get obstacle measurement
            # get odometry as change from previous pos
            
            # kalman predict from odometry -> mu_t
            # and obstacles -> obstacle measurement after the move -> mu_t
            
            # update path
            
            
            next_move = path_execution.get_current_move()
            turtle.cmd_velocity(angular = next_move[1], linear = next_move[0])
            # move change, record time from here
            previous_time = time.time()
            rate.sleep()

            if next_move[0] == 0 and next_move[1] == 0:
                print("Robot has stopped, making a measurement")
                time.sleep(0.5)
                image = turtle.get_rgb_image()
                pc_image = turtle.get_point_cloud()

                rectg_processor = RectangleProcessor(image,
                                                pc_image,
                                                color_settings)
                detected_rectgs, masked_rectgs, image_rectg  = rectg_processor.process_image()
                obstacle_measurements = detected_rectgs
            else:
                obstacle_measurements = []

            current_odometry = turtle.get_odometry()
            odometry_change = current_odometry - previous_odometry
            # start recording odometry changes right after using it
            previous_odometry = current_odometry

            # need to know the time here, so hope the rest of the cycle
            # until previous time update takes very little time
            # |
            # V lines should be very quick, robot is moving but it's not accounted for in the kalman
            # TODO Check how much of a problem that is
            
            # Or maybe don't need it at all
            # dt = time.time() - previous_time
            
            # update mu_t-1 to mu_t
            kalman_filter.pos_update(odometry_change)
            
            # measurements are from mu_t
            kalman_filter.obstacles_measurement_update(obstacle_measurements)
            kalman_filter.update_for_visualization()
            
            path_execution.update_path()
            
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
            if counter > 20:
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
                
                env.update_checkpoints()
                
                path_execution.update_path()
                previous_time = time.time()

                
        # rgb = turtle.get_rgb_image()
        # cv2.imshow('RGB Camera', rgb)     
         
        counter += 1
        
            
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