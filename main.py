# import math
# import numpy as np
import cv2
import time

# Running pygame without the display to resolve a dependency with OpenGL
import pygame 
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from visualization import RobotVisualization
from environment import Environment, Robot, Checkpoint, Obstacle

# from robolab_turtlebot import Turtlebot, get_time, Rate

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
    #                   [Checkpoint(30, 0, 0)],
    #                   set())

    env = Environment(Robot(0, 0, 0), 
                        [Checkpoint(100, 0, 0)],
                        {Obstacle(50,-20),
                         Obstacle(50,20)})
    
    screen_dimensions = (2160, 3840)
    # screen_dimensions = (1440, 900)
    vis = RobotVisualization(env, screen_dimensions)
    
    running = True
    counter = 0
    previous_time = time.time()
    while running:          
        # if counter % 10 == 0:
        vis.screen.fill((0, 0, 0))
        vis.draw_everything()
        if env.path:
            vis.draw_path(env.path)
        
        # print("Odometry:", odometry)
        # print("Robot position:", env.robot.x, env.robot.y, env.robot.a)
        vis.show_cv2()
        # counter += 1

        # rgb = turtle.get_rgb_image()
        # cv2.imshow('RGB Camera', rgb)
        
        # if (cv2.waitKey(1) & 0xFF == ord('q')) or turtle.is_shutting_down():
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            cv2.destroyAllWindows()
            pygame.quit()
            break
        next_move = env.get_current_move()
        print(next_move)
        env.simulate_movement(next_move, max(0, time.time() - previous_time))
        previous_time = time.time()
        vis.clock.tick(120)

        # turtle.cmd_velocity(angular = next_move[1], linear = next_move[0])
        # rate.sleep()
            
    print("Cycles:", counter)
        

if __name__ == "__main__":
    main()    