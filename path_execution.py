import numpy as np

def distance(point1, point2) -> float:
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

class PathExecution:
    def __init__(self, env) -> None:
        self.env = env
        self.total_distance_to_node = 0
        self.current_next_node = None
        
 
        
    def calculate_linear_speed(self, distance_to_node):
        acceleration_distance = self.env.robot.linear_acceleration_time * self.env.robot.linear_speed
        steepness = 1 / acceleration_distance
        
        scale = min(max(0, steepness * distance_to_node), 1) \
            - min(max(0, steepness * (distance_to_node-self.total_distance_to_node) + 1),1)
        return max(self.env.robot.minimal_linear_speed, self.env.robot.linear_speed * scale)
    
    def calculate_angular_speed_with_min(self, angle_difference):
        direction =  np.sign(angle_difference)
        acceleration_distance = self.env.robot.angular_acceleration_time * self.env.robot.angular_speed
        steepness = 1 / acceleration_distance
        
        scale = min(max(0, steepness*abs(angle_difference)), 1)
        speed = self.env.robot.angular_speed * scale
        speed = max(self.env.robot.minimal_angular_speed, speed)
        return speed * direction
    
    def calculate_angular_speed(self, angle_difference):
        direction =  np.sign(angle_difference)
        acceleration_distance = self.env.robot.angular_acceleration_time * self.env.robot.angular_speed
        steepness = 1 / acceleration_distance

        scale = min(max(0, steepness * abs(angle_difference)), 1)
        return self.env.robot.angular_speed * scale * direction
    
    def move_through_path(self):
        distance_to_node = distance(self.env.robot, self.current_next_node)
        if not self.current_next_node:
            return (0, 0)
        if distance_to_node > self.env.robot.node_distance_allowance:
            dx = self.current_next_node.x - self.env.robot.x
            dy = self.current_next_node.y - self.env.robot.y
            target_angle = np.arctan2(dy, dx)
            angle_difference = target_angle - self.env.robot.a
            angle_difference = np.arctan2(np.sin(angle_difference), np.cos(angle_difference))
            
            if abs(angle_difference) >= self.env.robot.node_angle_allowance:
                angular_speed = self.calculate_angular_speed_with_min(angle_difference)
                return (0, angular_speed)
            else:
                linear_speed = self.calculate_linear_speed(distance_to_node)
                angular_speed = self.calculate_angular_speed(angle_difference)
                return (linear_speed, angular_speed)
        else:
            # only one node left, probably checkpoint
            if len(self.env.path) == 1:
                angle_difference = self.current_next_node.a - self.env.robot.a
                angle_difference = np.arctan2(np.sin(angle_difference), np.cos(angle_difference))
                angular_speed = self.calculate_angular_speed_with_min(angle_difference)
                if abs(angle_difference) >= self.env.robot.node_angle_allowance:
                    return (0, angular_speed)
                self.env.current_goal_checkpoint_index += 1
            
            self.env.path.pop(0)
            self.update()
            return (0, 0)
        
    def update(self):
        if not self.env.path:
            # print("Trying to update next node for path execution, but the path is empty")
            return
        self.current_next_node = self.env.path[0]
        self.total_distance_to_node = distance(self.env.robot, self.current_next_node)

            