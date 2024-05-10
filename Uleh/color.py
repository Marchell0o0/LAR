import cv2 # type: ignore
import numpy as np # type: ignore
# import matplotlib.pyplot as plt # type: ignore
import math
import time
from collections import deque
from scipy.stats import norm # type: ignore
from robolab_turtlebot import Rate

import Uleh.utils
from Uleh.hue import calculate_hue_params
from Uleh.saturation import calculate_saturation_threshold, calculate_saturation_threshold_green

class ColorSettings:
    def __init__(
        self, 
        blue_range=[70, 130],
        # was blue_deviation=50 01.05.2024
        blue_deviation=70,
        green_range=[35, 65],
        # was green_deviation=70 01.05.2024
        green_deviation=50,
        # was red_range=[0, 4] 10.05.2024
        red_range=[0, 4],
        # was red_deviation=70 10.05.2024
        red_deviation=70,
        saturation_range=[20, 235]
        ) -> None:
        self.blue_range = blue_range
        self.blue_deviation = blue_deviation
        self.green_range = green_range
        self.green_deviation = green_deviation
        self.red_range = red_range
        self.red_deviation = red_deviation
        self.saturation_range = saturation_range
        self.colors = ["blue", "green", "red"]
        self.measured_values = {
            color_name: {
                "hue_values": [],
                "hue_deviations": [],
                "saturation_thresholds": []
                } for color_name in self.colors}
        self.calib_values = {
            color_name: {
                    "hue_avg": 0,
                    "dev_avg": 0,
                    "sat_avg": 0,
                    "color_reassigned": False
                    } for color_name in self.colors}
    
    def calculate_color_thresholds(self, image, color_name):
        # Access color-specific settings
        color_range = getattr(self, f"{color_name}_range")
        deviation_range = getattr(self, f"{color_name}_deviation")
        saturation_range = self.saturation_range
        # TODO: remove hard code for a green color
        if color_name == "green":
            saturation_threshold = calculate_saturation_threshold_green(image, saturation_range)
        else:
            saturation_threshold = calculate_saturation_threshold(image, saturation_range, color_name)
        hue_value, hue_deviation, hist_hue_smoothed = calculate_hue_params(
            image, 
            color_range,
            deviation_range,
            saturation_threshold,
            color_name
        )
        
        # plot_histogram(hist_hue_smoothed, "hue")
        
        return hue_value, hue_deviation, saturation_threshold
        
    def calibrate_color_debug(self, images, image=None):
        for image_path in images:
            image = np.load(image_path)
            for color_name in self.colors:
                
                hue_value, hue_deviation, saturation_threshold = self.calculate_color_thresholds(image, color_name)
                
                self.measured_values[color_name]["hue_values"].append(hue_value)
                self.measured_values[color_name]["hue_deviations"].append(hue_deviation)
                self.measured_values[color_name]["saturation_thresholds"].append(saturation_threshold)
        
        # Calculate averages
        self.assign_calibration_values()
        return
    
    def watch_environment(self, image):
        for color_name in self.colors:
            hue_value, hue_deviation, saturation_threshold = self.calculate_color_thresholds(image, color_name)
            
            self.measured_values[color_name]["hue_values"].append(hue_value)
            self.measured_values[color_name]["hue_deviations"].append(hue_deviation)
            self.measured_values[color_name]["saturation_thresholds"].append(saturation_threshold)  
        return    
    
    def save_image_values(self, turtle):
        
        image = turtle.get_rgb_image()
        
        if image is not None:
            for color_name in self.colors:
                
                    hue_value, hue_deviation, saturation_threshold = self.calculate_color_thresholds(image, color_name)
                    
                    if not np.isnan(hue_value) and not np.isnan(hue_deviation) and not np.isnan(saturation_threshold):
                        self.measured_values[color_name]["hue_values"].append(hue_value)
                        self.measured_values[color_name]["hue_deviations"].append(hue_deviation)
                        self.measured_values[color_name]["saturation_thresholds"].append(saturation_threshold)
        return
    
    def assign_calibration_values(self):
        for color_name in self.colors:
            avg_hue = np.mean(self.measured_values[color_name]["hue_values"])
            avg_deviation = np.mean(self.measured_values[color_name]["hue_deviations"])
            avg_saturation = np.mean(self.measured_values[color_name]["saturation_thresholds"])
            if not np.isnan(avg_hue) and not np.isnan(avg_deviation) and not np.isnan(avg_saturation):
                self.calib_values[color_name]["hue_avg"] = int(avg_hue)
                self.calib_values[color_name]["dev_avg"] = int(avg_deviation)
                self.calib_values[color_name]["sat_avg"] = int(avg_saturation)
                print(f"Average for {color_name}: Hue={int(avg_hue)}, Deviation={int(avg_deviation)}, Saturation Threshold={int(avg_saturation)}")
            else:
                raise ValueError("Unable to calibrate the camera: either avg_hue or avg_deviation or avg_saturation in NaN")
        return 
    
    # USE FOR A ROBOT
    def calibrate_color(self, turtle):
        # TODO: check if this Rate is alright
        rate = Rate(100)
        
        # Define a small threshold (epsilon), for example, 0.1
        epsilon = 0.03
        num_turns = 2
        
        for _ in range(num_turns):
            turtle.reset_odometry()
            time.sleep(0.1)
            a = turtle.get_odometry()[2]
            # Keeps track of the last multiple for which the condition was met
            last_triggered_multiple = None
            time.sleep(0.1)
            while abs(abs(a) - math.pi) >= epsilon and not turtle.is_shutting_down():
                turtle.cmd_velocity(angular=math.pi/2)
                
                # Calculate the nearest multiple of π/6 to 'a'
                nearest_multiple = round(a / (math.pi / 6)) * (math.pi / 6)

                # Check if 'a' is within epsilon of the nearest multiple of π/6
                if abs(a - nearest_multiple) < epsilon:
                    if nearest_multiple != last_triggered_multiple:
                        print("Triggered at a multiple of π/6:", a)
                        turtle.cmd_velocity(angular=0)
                        time.sleep(0.3)
                        
                        # ------Function to call------
                        self.save_image_values(turtle)
                        # ---------------------------
                        
                        turtle.cmd_velocity(angular=math.pi/2)
                        last_triggered_multiple = nearest_multiple

                rate.sleep()
                a = turtle.get_odometry()[2]
            
            turtle.cmd_velocity(angular=0)
            turtle.reset_odometry()
            time.sleep(0.5)
        
        self.assign_calibration_values()
        return    
    
    def calculate_rectangle_color(self, rectg, image, sat_offset=20, sat_deviation=100):
        cX, cY = rectg.cX, rectg.cY
        height, width, angle_rot = rectg.height, rectg.width, rectg.angle_rot
        points = Uleh.utils.calculate_rectangle_points(image, cX, cY, height, width, angle_rot)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_channel, saturation_channel, _ = cv2.split(hsv)
        
        # hue_channel = cv2.GaussianBlur(hue_channel, (7, 7), 0)
        
        color_name = Uleh.utils.color_value_to_str(rectg.color)
        color_deviation = self.calib_values[color_name]["dev_avg"]
        
        # print(f"Hue value in center for {color_name} is: {hue_channel[points[7][1], points[7][0]]}")
        # my_point = (613, 449)
        # # my_point = (618, 153)
        # # my_point = (652, 153)
        # # my_point = (648, 450)
        # print(f"Hue value at {my_point} for {color_name} is: {hue_channel[my_point[0], my_point[1]]}")
        # color_d = (0, 255, 0)
        # radius_d = 2  # Small radius for a dot-like appearance
        # thickness_d = 1  # Fill the circle
        # cv2.circle(image, my_point, radius_d, color_d, thickness_d)
        # cv2.imshow('Edge point:', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        hue_values_new = []
        sat_values_new = []
        for (px, py) in points:
            # Hue channel
            if np.abs(hue_channel[py, px] - self.calib_values[color_name]["hue_avg"]) < color_deviation:
            #    hue_dif = np.abs(hue_channel[py, px] - self.calib_values[color_name]["hue_avg"])
            #    print(f"HUE dif: {hue_dif}")
               hue_values_new.append(hue_channel[py, px]) 
            #    print(f"hue_value at {px, py} for {color_name} is: {hue_channel[py, px]}")
               
            # Saturation channel
            if np.abs(saturation_channel[py, px] - self.calib_values[color_name]["sat_avg"]) < sat_deviation:
            # sat_dif = np.abs(saturation_channel[py, px] - self.calib_values[color_name]["sat_avg"])
            # print(f"SAT difference: {sat_dif}")
                # print(np.abs(saturation_channel[py, px] - self.calib_values[color_name]["sat_avg"]))
                sat_values_new.append(saturation_channel[py, px] - sat_offset)
                # print(f"sat_value at {px, py} for {color_name} is: {saturation_channel[py, px]}")
               
        if len(hue_values_new) != 0 and len(sat_values_new) != 0:
            # print(hue_values_new)
            hue_value = int(np.mean(hue_values_new))
            sat_value = int(np.mean(sat_values_new))
        else:
            hue_value, sat_value = None, None
            
        return hue_value, sat_value 
    
    def reassign_first_color_values(self, detected_rectgs, image):
        # Assign color value corresponding to the color of found obstacle
        if (self.calib_values["blue"]["color_reassigned"] == False or
            self.calib_values["red"]["color_reassigned"] == False):
            
            for rectg in detected_rectgs:
                color_name = Uleh.utils.color_value_to_str(rectg.color)
                if self.calib_values[color_name]["color_reassigned"] == False:
                    hue_value, sat_value = self.calculate_rectangle_color(rectg, image)
                    hue_avg = self.calib_values[color_name]["hue_avg"]
                    sat_avg = self.calib_values[color_name]["sat_avg"]
                    if hue_value is not None and sat_value is not None:
                        self.calib_values[color_name]["hue_avg"] = hue_value
                        self.calib_values[color_name]["sat_avg"] = sat_value
                        print(f"New average HUE for {color_name}: {hue_avg}")
                        print(f"New average SAT for {color_name}: {sat_avg}")
        return
    
    # USE FOR A ROBOT
    def adapt_image_colors(self, turtle, color_settings, color_adapt_queue):
    # Assign color value corresponding to the color of found obstacle
        from Uleh.rectangle import RectangleProcessor
        rate = Rate(100)
        
        turtle.reset_odometry()
        a = turtle.get_odometry()[2]
        # Keeps track of the last multiple for which the condition was met
        last_triggered_multiple = None
        # Define a small threshold (epsilon), for example, 0.1
        epsilon = 0.03

        while ((self.calib_values["blue"]["color_reassigned"] == False or
            self.calib_values["red"]["color_reassigned"] == False) and
               not turtle.is_shutting_down()):
            
            turtle.cmd_velocity(angular=math.pi/2)
            
            # Calculate the nearest multiple of π/6 to 'a'
            nearest_multiple = round(a / (math.pi / 6)) * (math.pi / 6)

            # Check if 'a' is within epsilon of the nearest multiple of π/6
            if abs(a - nearest_multiple) < epsilon:
                if nearest_multiple != last_triggered_multiple:
                    print("Triggered at a multiple of π/6:", a)
                    turtle.cmd_velocity(angular=0)
                    time.sleep(0.3)
                    
                    # ------Function to call------
                    image = turtle.get_rgb_image()
                    pc_image = turtle.get_point_cloud()
                    
                    rectg_processor = RectangleProcessor(image,
                                                        pc_image,
                                                        color_settings,
                                                        color_adapt_queue)
                    _, _, _, detected_rectgs  = rectg_processor.process_image()
                    
                    for rectg in detected_rectgs:
                        color_name = Uleh.utils.color_value_to_str(rectg.color)
                        # To continue both blue and red color must be reassigned
                        if ((color_name == "blue" or color_name == "red") 
                            and not self.calib_values[color_name]["color_reassigned"]):
                            
                            self.calib_values[color_name]["color_reassigned"] = True
                            print("*****************************")
                            print(f"New average after REASSGINMENT for {color_name}: {self.calib_values[color_name]['hue_avg']}")
                            print("*****************************")
                    
                    # update_image_colors(
                    #     detected_rectgs,
                    #     image,
                    #     color_settings,
                    #     color_adapt_queue)
                    
                    # TODO: remove this old code    
                    # for rectg in detected_rectgs:
                    #     color_name = Uleh.utils.color_value_to_str(rectg.color)
                    #     # To continue both blue and red color must be reassigned
                    #     if ((color_name == "blue" or color_name == "red") 
                    #         and not self.calib_values[color_name]["color_reassigned"]):
                            
                    #         update_image_colors(rectgs, image, color_settings, color_adapt_queue)
                            
                    #         print("*****************************")
                    #         print(f"New average after REASSGINMENT for {color_name}: {self.calib_values[color_name]["hue_avg"]}")
                    # ---------------------------
                    
                    turtle.cmd_velocity(angular=math.pi/2)
                    last_triggered_multiple = nearest_multiple

            rate.sleep()
            a = turtle.get_odometry()[2]
        
        turtle.cmd_velocity(angular=0)
        turtle.reset_odometry()
        return    

class ColorHueQueue:
    def __init__(self, color_settings, max_length=10):
        self.max_length = max_length
        self.color_settings = color_settings
        self.queues = {}
        self.initialize()

    def initialize(self):
        """Initializes each color queue with a specified hue_avg value repeated."""
        for color_name in self.color_settings.colors:
            initial_hue_avg = self.color_settings.calib_values[color_name]["hue_avg"]
            self.queues[color_name] = deque(
                [initial_hue_avg] * self.max_length,
                maxlen=self.max_length)
            print(self.queues[color_name])

    def add(self, color_name, hue_value):
        """Adds a new hue value to the specified color queue."""
        if color_name in self.queues:
            self.queues[color_name].append(hue_value)
        else:
            raise ValueError("Invalid color name")

    def calculate_weights(self, color_name):
        """Generates weights with a Gaussian-like distribution for the specified color."""
        queue = self.queues[color_name]
        num_elements = len(queue)
        if num_elements == 0:
            return []

        x = np.linspace(-3, 3, num_elements)
        weights = norm.pdf(x)
        # Normalize such that the max weight is not less than 0.5
        weights /= weights.max() / 0.5  
        return weights

    def weighted_average(self, color_name):
        """Calculates the weighted average of the hue values in the specified color queue."""
        if color_name not in self.queues:
            raise ValueError("Invalid color name")

        weights = self.calculate_weights(color_name)
        queue = self.queues[color_name]

        if len(weights) != len(queue):
            raise ValueError("The number of weights must match the number of hue values")

        return np.average(queue, weights=weights)
    
    def average(self, color_name):
        """Calculates the average of the hue values in the specified color queue."""
        if color_name not in self.queues:
            raise ValueError("Invalid color name")

        queue = self.queues[color_name]
        
        return np.mean(queue)
    
    def adjust_detected_colors(self, rectgs, image, color_name):
        new_hue_value = None
        new_sat_value = None
        for rectg in rectgs:
            rectg_color = Uleh.utils.color_value_to_str(rectg.color)
            if rectg_color == color_name:
                rectg_hue, new_sat_value = self.color_settings.calculate_rectangle_color(rectg, image)
                if rectg_hue is not None:
                    color_deviation = self.color_settings.calib_values[color_name]["dev_avg"]
                    
                    # calculate some smart lower threshold
                    # for a new hue value to be added
                    lower_threshold = color_deviation // 6
                    
                    hue_dif = np.abs(rectg_hue - self.color_settings.calib_values[color_name]["hue_avg"])
                    # print(f"HUE dif for {color_name} and with {lower_threshold} lower_threshold: {hue_dif}")
                    if (np.abs(hue_dif) > lower_threshold and
                        np.abs(hue_dif) < color_deviation):
                        self.add(color_name, rectg_hue)
                        # self.color_settings.calib_values[color_name]["hue_avg"] = int(self.average(color_name))
                        old_hue = self.color_settings.calib_values[color_name]["hue_avg"]
                        new_hue_value = int(self.weighted_average(color_name))
                        # if old_hue != new_hue_value:
                            # print("***********")
                            # print(f"New QUEUE average HUE for {color_name}: {new_hue_value}")
                            # print("***********")
        return new_hue_value, new_sat_value

def update_image_colors(rectgs, image, color_settings, color_adapt_queue):
    for color_name in color_settings.colors:
        
        env_hue, env_dev, env_sat = color_settings.calculate_color_thresholds(image, color_name)
                    
        color_settings.measured_values[color_name]["hue_values"].append(env_hue)
        color_settings.measured_values[color_name]["hue_deviations"].append(env_dev)
        color_settings.measured_values[color_name]["saturation_thresholds"].append(env_sat)
        
        env_avg_hue = np.mean(color_settings.measured_values[color_name]["hue_values"])
        env_avg_dev = np.mean(color_settings.measured_values[color_name]["hue_deviations"])
        env_avg_sat = np.mean(color_settings.measured_values[color_name]["saturation_thresholds"])
        rectg_hue, rectg_sat = color_adapt_queue.adjust_detected_colors(rectgs, image, color_name)
        
        if rectg_hue is not None:
            old_hue = color_settings.calib_values[color_name]["hue_avg"]
            
            new_hue_value = int(np.mean([env_avg_hue, rectg_hue]))
            
            if old_hue != new_hue_value:
                color_settings.calib_values[color_name]["hue_avg"] = new_hue_value
                # print("|||||||||||")
                # print(f"New GENERAL average HUE for {color_name}: {new_hue_value}")
                # print("|||||||||||")
        elif rectg_hue is None:
            old_hue = color_settings.calib_values[color_name]["hue_avg"]
            new_hue_value = int(env_avg_hue)
            
            if old_hue != new_hue_value:
                color_settings.calib_values[color_name]["hue_avg"] = new_hue_value
                # print("|||||||||||")
                # print("rectg_hue is None")
                # print(f"New GENERAL average HUE for {color_name}: {new_hue_value}")
                # print("|||||||||||")
        old_dev = color_settings.calib_values[color_name]["dev_avg"]
        new_dev_value = int(env_avg_dev)
        if old_dev != new_dev_value: 
            color_settings.calib_values[color_name]["dev_avg"] = new_dev_value
            # print("______________")
            # print(f"New DEVIATION average for {color_name}: {new_dev_value}")
            # print("______________")
        if rectg_sat is not None:
            old_sat = color_settings.calib_values[color_name]["sat_avg"]
            new_sat_value = int(rectg_sat)
            if old_sat != new_sat_value:
                color_settings.calib_values[color_name]["sat_avg"] = new_sat_value
                # print("///////////")
                # print(f"New SATURATION average for {color_name}: {new_sat_value}")
                # print("///////////")
    return     
        
def create_mask(image, color_settings, color_name, color_params):
    
    hue_value = color_params[color_name]["hue_avg"]
    hue_deviation = color_params[color_name]["dev_avg"]
    saturation_threshold = color_params[color_name]["sat_avg"]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_channel, saturation_channel, value_channel = cv2.split(hsv)
         
    # Handling wrap-around for hue values near the boundaries
    lower_bound = hue_value - hue_deviation
    upper_bound = hue_value + hue_deviation

    if lower_bound < 0:
        # Mask for lower wrap-around
        mask1 = (hue_channel >= (180 + lower_bound)) & (hue_channel <= 179)
        # Mask for upper part of the range
        mask2 = (hue_channel >= 0) & (hue_channel <= upper_bound)
        masked_hue = np.where(mask1 | mask2, 255, 0).astype(np.uint8)
    elif upper_bound > 179:
        # Mask for upper wrap-around
        mask1 = (hue_channel >= lower_bound) & (hue_channel <= 179)
        # Mask for lower part of the range
        mask2 = (hue_channel >= 0) & (hue_channel <= upper_bound - 180)
        masked_hue = np.where(mask1 | mask2, 255, 0).astype(np.uint8)
    else:
        # Normal case, no wrap-around
        masked_hue = np.where((hue_channel >= lower_bound) & (hue_channel <= upper_bound), 255, 0).astype(np.uint8)
    masked_sat = np.where((saturation_channel > saturation_threshold), 255, 0).astype(np.uint8)
    mask = np.where(((masked_hue == 255) & (masked_sat == 255)), 255, 0).astype(np.uint8)

    # Debugging: 
    #Original:
    # cv2.imshow('Original', image)
    # cv2.imshow('Hue', hue_visualized)
    # cv2.imshow('Saturation', saturation_channel)
    # cv2.imshow('Value', value_channel)
    
        
    # Hue, Saturation, Value masked: 
    # if color_name == "red":
        # cv2.imshow('Masked sat', masked_sat)
        # cv2.imshow('Masked hue', masked_hue)
    # cv2.imshow('Mask', mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    return mask

# TODO: create a class for color_value assignment
def merge_masks(blue_mask, green_mask, red_mask):
    combined_mask = np.zeros_like(blue_mask)
    combined_mask[(blue_mask == 255) & (combined_mask == 0)] = Uleh.utils.str_to_color_value("blue")
    combined_mask[(green_mask == 255) & (combined_mask == 0)] = Uleh.utils.str_to_color_value("green")
    combined_mask[red_mask == 255] = Uleh.utils.str_to_color_value("red")
    
    # Debugging
    # cv2.imshow('Combined mask', combined_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return combined_mask


