# import numpy as np
import math


def distance(point1, point2) -> float:
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

class PathExecution:
    def __init__(self, env) -> None:
        self.env = env
        self.total_distance_to_node = 0
        self.current_next_node = None
        
        self.distance_allowance = 1
        self.angle_allowance = 1
        self.linear_acceleration_time = 1
        self.angular_acceleration_time = 1
        self.minimal_linear_speed = 0.1 # m/s
        self.minimal_angular_speed = 0.3 # rad/s
        
    def calculate_linear_speed(self, distance_to_node):
        acceleration_distance = self.linear_acceleration_time * self.env.robot.linear_speed * 100 # cm
        scale = min(max(0, (1/acceleration_distance)*distance_to_node), 1) \
            - min(max(0, (1/acceleration_distance)*(distance_to_node-self.total_distance_to_node)+1),1)
        return max(self.minimal_linear_speed, self.env.robot.linear_speed * scale)
    
    def calculate_angular_speed_with_min(self, angle_difference):
        direction =  (angle_difference > 0) - (angle_difference < 0)
        print(direction)
        acceleration_distance = (self.angular_acceleration_time * self.env.robot.angular_speed)
        scale = min(max(0, (1/acceleration_distance)*abs(angle_difference)), 1)
        speed = self.env.robot.angular_speed * scale
        speed = max(self.minimal_angular_speed, speed)
        return speed * direction
    
    def calculate_angular_speed(self, angle_difference):
        direction =  (angle_difference > 0) - (angle_difference < 0)
        print(direction)
        acceleration_distance = (self.angular_acceleration_time * self.env.robot.angular_speed)
        scale = min(max(0, (1/acceleration_distance)*abs(angle_difference)), 1)
        return self.env.robot.angular_speed * scale * direction
    
    def move_through_path(self):
        distance_to_node = distance(self.env.robot, self.current_next_node)
        if not self.current_next_node:
            return (0, 0)
        if distance_to_node > self.distance_allowance:
            dx = self.current_next_node.x - self.env.robot.x
            dy = self.current_next_node.y - self.env.robot.y
            
            target_angle = math.degrees(math.atan2(dy, dx))
            angle_difference = (target_angle - self.env.robot.a + 180) % 360 - 180
            print(angle_difference)
            if abs(angle_difference) >= self.angle_allowance:
                angular_speed = self.calculate_angular_speed_with_min(math.radians(angle_difference))
                return (0, angular_speed)
            else:
                linear_speed = self.calculate_linear_speed(distance_to_node)
                angular_speed = self.calculate_angular_speed(math.radians(angle_difference))
                return (linear_speed, angular_speed)
        else:
            # only one node left, probably checkpoint
            if len(self.env.path) == 1:
                angle_difference = (self.current_next_node.a - self.env.robot.a + 180) % 360 - 180
                angular_speed = self.calculate_angular_speed_with_min(math.radians(angle_difference))
                if abs(angle_difference) >= self.angle_allowance:
                    return (0, angular_speed)
            
            self.env.path.pop(0)
            self.env.current_goal_checkpoint_index += 1
            self.update()
            return (0, 0)
        
    def update(self):
        if not self.env.path:
            # print("Trying to update next node for path execution, but the path is empty")
            return
        self.current_next_node = self.env.path[0]
        self.total_distance_to_node = distance(self.env.robot, self.current_next_node)

            