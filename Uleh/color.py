import cv2 # type: ignore
import numpy as np # type: ignore
# import matplotlib.pyplot as plt # type: ignore
import math
import time

import Uleh.utils
from Uleh.hue import calculate_hue_params
from Uleh.saturation import calculate_saturation_threshold, calculate_saturation_threshold_green
from robolab_turtlebot import Rate

class ColorSettings:
    def __init__(
        self, 
        blue_range=[70, 130],
        # was blue_deviation=50 01.05.2024
        blue_deviation=70,
        green_range=[35, 65],
        # was green_deviation=70 01.05.2024
        green_deviation=50,
        red_range=[0, 4],
        # was red_deviation=50 01.05.2024
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
        
    def calibrate_color_debug(self, images):
        color_averages = {
            color_name: {
                "hue_values": [],
                "hue_deviations": [],
                "saturation_thresholds": []
                } for color_name in self.colors}
        
        for image_path in images:
            image = np.load(image_path)
            for color_name in self.colors:
                
                hue_value, hue_deviation, saturation_threshold = self.calculate_color_thresholds(image, color_name)
                
                color_averages[color_name]["hue_values"].append(hue_value)
                color_averages[color_name]["hue_deviations"].append(hue_deviation)
                color_averages[color_name]["saturation_thresholds"].append(saturation_threshold)
        
        # Calculate averages
        for color_name in self.colors:
            avg_hue = np.mean(color_averages[color_name]["hue_values"])
            avg_deviation = np.mean(color_averages[color_name]["hue_deviations"])
            avg_saturation = np.mean(color_averages[color_name]["saturation_thresholds"])
            self.calib_values[color_name]["hue_avg"] = int(avg_hue)
            self.calib_values[color_name]["dev_avg"] = int(avg_deviation)
            self.calib_values[color_name]["sat_avg"] = int(avg_saturation)
            print(f"Average for {color_name}: Hue={int(avg_hue)}, Deviation={int(avg_deviation)}, Saturation Threshold={int(avg_saturation)}")
        return
    
    def save_image_values(self, turtle, color_data):
        
        image = turtle.get_rgb_image()
        
        if image:
            for color_name in self.colors:
                
                    hue_value, hue_deviation, saturation_threshold = self.calculate_color_thresholds(image, color_name)
                    
                    if not np.isnan(hue_value) and not np.isnan(hue_deviation) and not np.isnan(saturation_threshold):
                        color_data[color_name]["hue_values"].append(hue_value)
                        color_data[color_name]["hue_deviations"].append(hue_deviation)
                        color_data[color_name]["saturation_thresholds"].append(saturation_threshold)
        return

    def calculate_color_averages(self, color_data):
        for color_name in self.colors:
            avg_hue = np.mean(color_data[color_name]["hue_values"])
            avg_deviation = np.mean(color_data[color_name]["hue_deviations"])
            avg_saturation = np.mean(color_data[color_name]["saturation_thresholds"])
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
        
        turtle.reset_odometry()
        a = turtle.get_odometry()[2]
        # Keeps track of the last multiple for which the condition was met
        last_triggered_multiple = None
        # Define a small threshold (epsilon), for example, 0.1
        epsilon = 0.03
        
        color_data = {
                color_name: {
                    "hue_values": [],
                    "hue_deviations": [],
                    "saturation_thresholds": []
                    } for color_name in self.colors}
        
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
                    self.save_image_values(turtle, color_data)
                    # ---------------------------
                    
                    turtle.cmd_velocity(angular=math.pi/2)
                    last_triggered_multiple = nearest_multiple

            rate.sleep()
            a = turtle.get_odometry()[2]
            # print(a)
        
        turtle.cmd_velocity(angular=0)
        turtle.reset_odometry()
        
        self.calculate_color_averages(color_data)
        return
    
    def assign_detected_color(self, rectg, image):
        cX, cY = rectg.cX, rectg.cY
        height, width, angle_rot = rectg.height, rectg.width, rectg.angle_rot
        points = Uleh.utils.calculate_rectangle_points(cX, cY, height, width, angle_rot)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_channel, saturation_channel, _ = cv2.split(hsv)
        
        hue_channel = cv2.GaussianBlur(hue_channel, (7, 7), 0)
        
        color_name = Uleh.utils.color_value_to_str(rectg.color)
        color_deviation = getattr(self, f"{color_name}_deviation")
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
            #    print(np.abs(hue_channel[py, px] - self.calib_values[color_name]["hue_avg"]))
               hue_values_new.append(hue_channel[py, px]) 
               print(f"hue_value at {px, py} for {color_name} is: {hue_channel[py, px]}")
               
            # Saturation channel
            # TODO: remove hard code
            if np.abs(saturation_channel[py, px] - self.calib_values[color_name]["sat_avg"]) < 50:
                sat_values_new.append(saturation_channel[py, px] - 30)
                print(f"sat_value at {px, py} for {color_name} is: {saturation_channel[py, px]}")
               
        if len(hue_values_new) != 0 and len(sat_values_new) != 0:
            # print(hue_values_new)
            self.calib_values[color_name]["hue_avg"] = int(np.mean(hue_values_new))
            self.calib_values[color_name]["sat_avg"] = int(np.mean(sat_values_new))
            self.calib_values[color_name]["color_reassigned"] = True
        return 
    
    # USE FOR A ROBOT
    def adapt_image_colors(self, turtle):
    # Assign color value corresponding to the color of found obstacle
        # TODO: check wether its okay to import like this
        from rectangle import RectangleProcessor
        rate = Rate(100)
        
        turtle.reset_odometry()
        a = turtle.get_odometry()[2]
        # Keeps track of the last multiple for which the condition was met
        last_triggered_multiple = None
        # Define a small threshold (epsilon), for example, 0.1
        epsilon = 0.03

        colors_reassigned_counter = 0
    
        while colors_reassigned_counter < 2 and not turtle.is_shutting_down():
            
            # Calculate the nearest multiple of π/6 to 'a'
            nearest_multiple = round(a / (math.pi / 6)) * (math.pi / 6)

            # Check if 'a' is within epsilon of the nearest multiple of π/6
            if abs(a - nearest_multiple) < epsilon:
                if nearest_multiple != last_triggered_multiple:
                    print("Triggered at a multiple of π/6:", a)
                    turtle.cmd_velocity(angular=0)
                    time.sleep(0.5)
                    
                    # ------Function to call------
                    image = turtle.get_rgb_image()
                    pc_image = turtle.get_point_cloud()
                    
                    rectg_processor = RectangleProcessor(image,
                                                        pc_image,
                                                        self.color_settings)
                    detected_rectgs, _, _  = rectg_processor.process_image()
                    
                    for rectg in detected_rectgs:
                        color_name = Uleh.utils.color_value_to_str(rectg.color)
                        # To continue both blue and red color must be reassigned
                        if ((color_name == "blue" or color_name == "red") 
                            and not self.calib_values[color_name]["color_reassigned"]):
                            # turtle.cmd_velocity(angular=0)
                            # time.sleep(0.5)
                            self.assign_detected_color(rectg, image)
                            colors_reassigned_counter += 1
                            print("*****************************")
                            print(f"New average for {color_name}: {self.calib_values[color_name]['hue_avg']}")
                    # ---------------------------
                    
                    turtle.cmd_velocity(angular=math.pi/6)
                    last_triggered_multiple = nearest_multiple

            rate.sleep()
            a = turtle.get_odometry()[2]
            # print(a)
        print("*****************************")
        
        turtle.cmd_velocity(angular=0)
        return    
        
def create_mask(image, color_settings, color_name, color_params):
    
    hue_value = color_params[color_name]["hue_avg"]
    hue_deviation = color_params[color_name]["dev_avg"]
    saturation_threshold = color_params[color_name]["sat_avg"]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_channel, saturation_channel, value_channel = cv2.split(hsv)

    # TODO: remove hard code for a green
    # if color_name == "green":
    #     masked_hue = calculate_hue_params_green(image,
    #                                             color_settings.blue_range,
    #                                             )
    # else:
    #     masked_hue = np.where((np.abs(hue_channel - hue_value) < hue_deviation), 255, 0).astype(np.uint8)
         
    # Create masks
    masked_hue = np.where((np.abs(hue_channel - hue_value) < hue_deviation), 255, 0).astype(np.uint8)
    masked_sat = np.where((saturation_channel > saturation_threshold), 255, 0).astype(np.uint8)
    mask = np.where(((masked_hue == 255) & (masked_sat == 255)), 255, 0).astype(np.uint8)

    # Debugging: 
    #Original:
    # cv2.imshow('Original', image)
    # cv2.imshow('Hue', hue_visualized)
    # cv2.imshow('Saturation', saturation_channel)
    # cv2.imshow('Value', value_channel)
    
        
    # Hue, Saturation, Value masked: 
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
