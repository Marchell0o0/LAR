import cv2
import time
import cProfile 
import pstats
import numpy as np

from visualization import RobotVisualization
from environment import Environment, Robot, Checkpoint, Obstacle
from path_execution import PathExecution
from path_creation import PathCreation
from kalman import KalmanFilter
# from robolab_turtlebot import Turtlebot, get_time, Rate

# Running pygame without the display to resolve a dependency with OpenGL
import pygame 
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

def main():

    # turtle = Turtlebot(rgb = True, depth = True, pc = True)
    
    # turtle = Turtlebot()

    # turtle.wait_for_rgb_image()
    # print('Rgb image received')
    
    # turtle.wait_for_point_cloud()
    # print('Point cloud received')
    
    # turtle.wait_for_depth_image()
    # print("Depth image received")
    
    # turtle.wait_for_odometry()
    # print("Odometry received")
    
    # turtle.reset_odometry()
    # odometry = turtle.get_odometry()
    
    # rate = Rate(100)
    
    # env = Environment(Robot(0, 0, 0), 
    #                     [Checkpoint(50, 50, 0),
    #                      Checkpoint(100, 0, 180),
    #                      Checkpoint(0, 0, 90)],
    #                     set())
    
    # env = Environment(Robot(0, 0, 0),
    #                   [Checkpoint(2, 0, 0)],
    #                   set(), set())   
     
    # env = Environment(Robot(0, 0, np.deg2rad(-180)),
    #                   [Checkpoint(2, 0, 0),
    #                    Checkpoint(0, 0, 0)],
    #                   [Obstacle(1, 0.2, "red"),
    #                    Obstacle(1, -0.2, "blue")], set())  
     

    env = Environment(Robot(-0.3, 0, 0), 
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
    
    screen_dimensions = (2160, 3840)
    # screen_dimensions = (1440, 900)
    vis = RobotVisualization(env, path_execution, kalman_filter, screen_dimensions)
    
    running = True
    counter = 0
    previous_time = 0
    # with cProfile.Profile() as pr:
    previous_obstacles_size = 0
    print("Starting main loop")
    while running:          
        # if counter % 10 == 0:
        vis.screen.fill((0, 0, 0))
        vis.draw_everything()

        vis.show_cv2()

        # odometry = turtle.get_odometry()
        # env.robot.x = odometry[0]
        # env.robot.y = odometry[1]
        # env.robot.a = odometry[2]
        # print("Odometry:", odometry)
        # print("Robot position:", env.robot.x, env.robot.y, env.robot.a)

        # rgb = turtle.get_rgb_image()
        # cv2.imshow('RGB Camera', rgb)      
          
        next_move = path_execution.get_current_move()
        # print(next_move)
    
            # print(env.path)
        
        
        # env.simulate_movement(next_move, max(0, time.time() - previous_time))
            
        if counter > 100:
            if previous_time == 0:
                previous_time = time.time()
            kalman_filter.process_measurement(next_move, measurements, time.time() - previous_time)
        previous_time = time.time()
            
        measurements = env.get_measurement()
        if len(env.obstacles) > previous_obstacles_size:
            previous_obstacles_size = len(env.obstacles)
            print("Resetting path")
            path_execution.path = []
            previous_time = 0
            # print("---------------------------------")
            # for obstacle in env.obstacles:
                # print(obstacle)
        vis.clock.tick(120)
        

        # turtle.cmd_velocity(angular = next_move[1], linear = next_move[0])
        # rate.sleep()
        
        # if (cv2.waitKey(1) & 0xFF == ord('q')) or turtle.is_shutting_down():
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
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