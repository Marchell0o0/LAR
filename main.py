import math
import cv2
import numpy as np
import time

# Running pygame without the display to resolve a dependency with OpenGL
import pygame 
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from a_star import A_star
from visualization import RobotVisualization
from environment import Environment, Robot, Checkpoint, Obstacle, Node

from robolab_turtlebot import Turtlebot, get_time, Rate

def reconstruct_path(came_from, goal):
    current = goal
    path = []
    while current is not None:  # Changed from `while current != start`
        path.append(current)
        current = came_from[current]
    path.reverse()  # reverse the path to start to goal
    return path

def compute_path(env, checkpoint_index):
    goal = env.checkpoints[checkpoint_index]
    # for goal in env.checkpoints:
        # list of nodes
    path = []
    if env.straight_path_exists(env.robot, goal):
        print("Found straight path")
        path = [env.robot, goal]
        return path
    else:
        print("Looking for path with A*")
        a_star = A_star(env)
        came_from, cost_so_far = a_star.search(goal)
        if goal in came_from:
            path = reconstruct_path(came_from, goal)
            print("Path found using A*")

    # if path:
    #     vis.draw_path(path)
    # else:
    #     print("No path found")
    
    simplified_path = env.simplify_path(path)
    if simplified_path:
        # print("Drawing simplified path...")
        for node in simplified_path:
            if not node in env.checkpoints:
                node.a = math.degrees(math.atan2(node.y - env.robot.y, node.x - env.robot.x))

        # env.robot.x = goal.x
        # env.robot.y = goal.y
    else:
        print("Couldn't simplify path")

    return simplified_path

def distance(point1: Node, point2: Node) -> float:
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

class PathExecution:
    def __init__(self, env, path) -> None:
        self.env = env
        self.path_index = 1
        self.path = path
        self.distance_allowance = 3
        
    def reset(self):
        self.path_index = 1
        if self.path:
            self.total_distance = distance(self.path[self.path_index - 1], self.path[self.path_index])
        
    def calculate_speed(self, distance_to_node, steepness = 1, time_for_acceleration = 0):
        offset_start = 0
        offset_end = 7
        scale = 1 / (1 + steepness * np.exp(offset_end-distance_to_node)) \
            - 1 / (1 + steepness * np.exp((self.total_distance - self.distance_allowance) - offset_start - distance_to_node))
        print("Scale and distances:", scale, distance_to_node, self.total_distance)
        return self.env.robot.linear_speed * scale
    
    def get_next_move(self, angle_allowance=5,):
        if not self.path:
            print("Path in execution is empty")
            return (-1, -1)
        if self.path_index >= len(self.path):
            print("No next node, arrived at checkpoint")
            return (-1, -1)

        # current_x = self.starting_coords[0] + odometry[0] * 100  # convert odometry to cm
        # current_y = self.starting_coords[1] + odometry[1] * 100
        # current_a = (self.starting_coords[2] + math.degrees(odometry[2]) + 180) % 360 - 180  # Normalize degrees

        # print("Current position POV robot:", current_x, current_y, current_a)

        next_node = self.path[self.path_index]
        dx = next_node.x - self.env.robot.x
        dy = next_node.y - self.env.robot.y
        target_angle = math.degrees(math.atan2(dy, dx))
        # print("Target calculations: dx =", dx, "dy =", dy, "Target angle =", target_angle)

        angle_difference = (target_angle - self.env.robot.a + 180) % 360 - 180  # Normalize angle
        # print("Angle difference:", angle_difference)
        distance_to_node = distance(Node(self.env.robot.x, self.env.robot.y), next_node)
        if  distance_to_node > self.distance_allowance:
            if abs(distance_to_node - self.total_distance) < 2 and abs(angle_difference) >= angle_allowance: # Haven't moved
                    turn_speed = -self.env.robot.angular_speed if angle_difference < 0 else self.env.robot.angular_speed
                    return (0, turn_speed)
            else:
                speed = self.calculate_speed(distance_to_node)
                turn_speed = 0
                if abs(angle_difference) >= angle_allowance:
                    turn_speed = -self.env.robot.angular_speed if angle_difference < 0 else self.env.robot.angular_speed
                # speed = self.env.robot.linear_speed * math.sin(math.pi * (distance_to_node / self.total_distance))
                # return (speed, 0)
                return (max(0.05, speed), turn_speed)
        else:
            if self.path_index == (len(self.path) - 1):  # Check if at the last node
                # print("At checkpoint, checking orientation...")
                final_angle_diff = (next_node.a - self.env.robot.a + 180) % 360 - 180
                if abs(final_angle_diff) >= angle_allowance:
                    turn_speed = -self.env.robot.angular_speed if final_angle_diff < 0 else self.env.robot.angular_speed
                    return (0, turn_speed)
            # Move to the next node in the path

            self.total_distance = distance(self.path[self.path_index - 1], next_node)
            self.path_index += 1
            return (0, 0)
            
            

        

def main():

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
    odometry = turtle.get_odometry()
    
    rate = Rate(100)
    
    env = Environment(Robot(0, 0, 0), 
                        [Checkpoint(50, 50, 0),
                         Checkpoint(100, 0, 180),
                         Checkpoint(0, 0, 90)],
                        set())
    
    # env = Environment(Robot(0, 0, 0),
    #                   [Checkpoint(30, 0, 0)],
    #                   set())

    # env = Environment(Robot(0, 0, 0), 
    #                     [Checkpoint(100, 0, 0)],
    #                     {Obstacle(50,-20),
    #                      Obstacle(50,20)})
    
    # screen_dimensions = (2160, 3840)
    screen_dimensions = (1440, 900)
    vis = RobotVisualization(env, screen_dimensions)
    
    checkpoint_index = 0
    path = []
    path_execution = PathExecution(env, path)

    running = True
    arrived = True # "Arrived to start"
    counter = 0
    while running:          
        if counter % 10 == 0:
            vis.screen.fill((0, 0, 0))
            vis.draw_everything()
            if path:
                vis.draw_path(path)
            
            print("Odometry:", odometry)
            print("Robot position:", env.robot.x, env.robot.y, env.robot.a)
            vis.show_cv2()
        counter += 1

        # rgb = turtle.get_rgb_image()
        # cv2.imshow('RGB Camera', rgb)
        
        if (cv2.waitKey(1) & 0xFF == ord('q')) or turtle.is_shutting_down():
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            cv2.destroyAllWindows()
            pygame.quit()
            break

        if arrived:
            # check for the next checkpoint
            if checkpoint_index < len(env.checkpoints):

                path = compute_path(env, checkpoint_index)
                path_execution.path = path
                path_execution.reset()

                old_position = (env.robot.x, env.robot.y, env.robot.a)
                turtle.reset_odometry()

                checkpoint_index += 1
                arrived = False
            else:
                # look for checkpoints(spin around)
                pass
        else:
            odometry = turtle.get_odometry()

            env.robot.x = math.cos(math.radians(old_position[2])) * odometry[0] * 100 + old_position[0]
            env.robot.y = math.cos(math.radians(old_position[2])) * odometry[1] * 100 + old_position[1]
            env.robot.a = (math.degrees(odometry[2]) + old_position[2] + 180) % 360 - 180
            
            next_move = path_execution.get_next_move()
            # print(next_move)
            if next_move[0] == 0 and next_move[1] == 0: # at a node
                pass
                # turtle.reset_odometry()
            elif next_move[0] == -1: # at checkpoint
                arrived = True
                continue
            else:
                # print(next_move)
                pass
                # odometry = env.simulate_movement(next_move, max(0, time.time() - previous_time), odometry)
                # print(odometry)
                # previous_time = time.time()

            turtle.cmd_velocity(angular = next_move[1], linear = next_move[0])
            

        rate.sleep()
        # vis.clock.tick(120)
    print("Cycles:", counter)
        

if __name__ == "__main__":
    main()    