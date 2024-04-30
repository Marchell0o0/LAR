import cv2
import time
import cProfile 
import pstats
import numpy as np

from visualization import RobotVisualization
from environment import Environment, Robot, Checkpoint, Obstacle
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
     

    env = Environment(Robot(0, 0, 0), 
                        [Checkpoint(1, 0, 0), Checkpoint(2, 1, np.pi/4)],
                        set(),
                        {Obstacle(0.50, -0.20, "red"),
                         Obstacle(0.50, 0.20, "blue"),
                         Obstacle(0.50, -0.60, "blue"),
                         Obstacle(1.5, 0.50, "blue"),
                         Obstacle(1.5, 0.75, "blue"),
                        #  Obstacle(1.75, 0.25, "blue", 5),
                        #  Obstacle(1.61, 0.43, "blue", 4),
                        #  Obstacle(1.43, 0.61, "blue", 3),
                        #  Obstacle(1.25, 0.75, "blue", 6)
                         })
    
    screen_dimensions = (2160, 3840)
    # screen_dimensions = (1440, 900)
    vis = RobotVisualization(env, screen_dimensions)
    
    running = True
    counter = 0
    previous_time = time.time()
    # with cProfile.Profile() as pr:
    previous_obstacles_size = 0
    while running:          
        # if counter % 10 == 0:
        vis.screen.fill((0, 0, 0))
        vis.draw_everything()
        if env.path:
            vis.draw_path(env.path)
        
        # print("Odometry:", odometry)
        # print("Robot position:", env.robot.x, env.robot.y, env.robot.a)
        vis.show_cv2()
        counter += 1

        # rgb = turtle.get_rgb_image()
        # cv2.imshow('RGB Camera', rgb)        
    
        
        measurements = env.get_measurement()
        
        if len(env.obstacles) > previous_obstacles_size:
            previous_obstacles_size = len(env.obstacles)
            print("Resetting path")
            env.reset_path()
            print(env.path)
        
        next_move = env.get_current_move()
        # env.simulate_movement(next_move, max(0, time.time() - previous_time))
        env.kalman_filter.process_measurement(next_move, measurements, time.time() - previous_time)
        
        previous_time = time.time()
        vis.clock.tick(120)

        # turtle.cmd_velocity(angular = next_move[1], linear = next_move[0])
        # rate.sleep()
        
        # if (cv2.waitKey(1) & 0xFF == ord('q')) or turtle.is_shutting_down():
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            
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