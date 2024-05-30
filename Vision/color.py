import cv2
import numpy as np
import math
import time
from collections import deque
from scipy.stats import norm

import Vision.utils
from Vision.hue import calculate_hue_params
from Vision.saturation import calculate_saturation_threshold, calculate_saturation_threshold_green

class ColorSettings:
    def __init__(
        self, 
        blue_range=[80, 130],
        blue_deviation=50,
        green_range=[50, 65],
        green_deviation=30,
        red_range=[0, 4],
        red_deviation=50,
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
                # default color params
                    "hue_avg": 100000,
                    "dev_avg": 10000,
                    "sat_avg": 100000,
                    "reassigned": False
                    } for color_name in self.colors}
    
    def calculate_color_thresholds(self, image, color_name, red_sat_offset=50):
        # Access color-specific settings
        color_range = getattr(self, f"{color_name}_range")
        deviation_range = getattr(self, f"{color_name}_deviation")
        saturation_range = self.saturation_range
        if color_name == "green":
            saturation_threshold = calculate_saturation_threshold_green(image, saturation_range)
        elif color_name == "red":
            saturation_threshold = calculate_saturation_threshold(image, saturation_range, color_name) - red_sat_offset
        elif color_name == "blue":
            saturation_threshold = calculate_saturation_threshold(image, saturation_range, color_name)
        hue_value, hue_deviation, _ = calculate_hue_params(
            image, 
            color_range,
            deviation_range,
            saturation_threshold,
            color_name
        )
        
        return hue_value, hue_deviation, saturation_threshold 
    
    def calculate_rectangle_color(self, rectg, image, sat_offset=40, sat_deviation=100):
        cX, cY = rectg.cX, rectg.cY
        height, width, angle_rot = rectg.height, rectg.width, rectg.angle_rot
        points = Vision.utils.calculate_rectangle_points(image, cX, cY, height, width, angle_rot)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_channel, saturation_channel, _ = cv2.split(hsv)
        
        color_name = Vision.utils.color_value_to_str(rectg.color)
        color_deviation = self.calib_values[color_name]["dev_avg"]
        
        hue_values_new = []
        sat_values_new = []
        for (px, py) in points:
            # Hue channel
            if np.abs(hue_channel[py, px] - self.calib_values[color_name]["hue_avg"]) < color_deviation:
               hue_values_new.append(hue_channel[py, px]) 
               
            # Saturation channel
            if np.abs(saturation_channel[py, px] - self.calib_values[color_name]["sat_avg"]) < sat_deviation:
                sat_values_new.append(saturation_channel[py, px])
               
        if len(hue_values_new) != 0 and len(sat_values_new) != 0:
            hue_value = int(np.mean(hue_values_new))
            sat_value = int(np.mean(sat_values_new) - sat_offset)
        else:
            hue_value, sat_value = None, None
            
        return hue_value, sat_value 
    
    def update_image_colors(self, rectgs, image, color_adapt_queue):
        for color_name in self.colors:
            env_hue, env_dev, env_sat = self.calculate_color_thresholds(image, color_name)
            self.prune_measured_values(color_name, max_elem=100)
            self.append_valid_measurements(color_name, env_hue, env_dev, env_sat)
            
            env_avg_hue, env_avg_dev, env_avg_sat = self.update_env_colors(color_name)

            self.update_calibration_value(color_adapt_queue, color_name, "dev", env_avg_dev)
            
            # Update hue based on presence of detected hue value
            rectg_hue, rectg_sat = color_adapt_queue.adjust_detected_colors(rectgs, image, color_name)
            if rectg_hue is not None:
                self.update_calibration_value(color_adapt_queue, color_name, "hue", rectg_hue)
            else:
                self.update_calibration_value(color_adapt_queue, color_name, "hue", env_avg_hue)

            # Update saturation based on presence of detected saturation value
            if rectg_sat is not None:
                self.update_calibration_value(color_adapt_queue, color_name, "sat", rectg_sat, multiplier=2)
            else:
                self.update_calibration_value(color_adapt_queue, color_name, "sat", env_avg_sat)
                
            if not self.calib_values[color_name]["reassigned"]:
                self.reassign_new_color(color_adapt_queue, color_name)
        # print(color_adapt_queue)
        return

    def prune_measured_values(self, color_name, max_elem=100):
        # Restrict the list of measured values to have max 100 elements 
        if (len(self.measured_values[color_name]["hue_values"]) > max_elem and
            len(self.measured_values[color_name]["hue_deviations"]) > max_elem and
            len(self.measured_values[color_name]["saturation_thresholds"]) > max_elem):
            
            num_elem = max_elem // 2
            
            del self.measured_values[color_name]["hue_values"][:num_elem]
            del self.measured_values[color_name]["hue_deviations"][:num_elem]
            del self.measured_values[color_name]["saturation_thresholds"][:num_elem]

    def append_valid_measurements(self, color_name, env_hue, env_dev, env_sat):
        if (not np.isnan(env_hue) and
            not np.isnan(env_dev) and
            not np.isnan(env_sat)):
                            
            self.measured_values[color_name]["hue_values"].append(env_hue)
            self.measured_values[color_name]["hue_deviations"].append(env_dev)
            self.measured_values[color_name]["saturation_thresholds"].append(env_sat)
        else:
            print("env_hue, env_dev or env_sat is NaN")

    def update_env_colors(self, color_name):
        if (len(self.measured_values[color_name]["hue_values"]) == 0 or
            len(self.measured_values[color_name]["hue_deviations"]) == 0 or
            len(self.measured_values[color_name]["saturation_thresholds"]) == 0):
            return
        
        env_avg_hue = np.mean(self.measured_values[color_name]["hue_values"])
        env_avg_dev = np.mean(self.measured_values[color_name]["hue_deviations"])
        env_avg_sat = np.mean(self.measured_values[color_name]["saturation_thresholds"])

        return env_avg_hue, env_avg_dev, env_avg_sat

    def update_calibration_value(self,
                                color_adapt_queue,
                                color_name,
                                value_type,
                                value,
                                use_detected_value=True,
                                multiplier=1):
        
        if use_detected_value:
            old_value = self.calib_values[color_name][f"{value_type}_avg"]
            for _ in range(multiplier):
                color_adapt_queue.add(color_name, value_type, int(value))
            new_value_avg = int(color_adapt_queue.weighted_average(color_name, value_type))
            if old_value != new_value_avg:
                self.calib_values[color_name][f"{value_type}_avg"] = new_value_avg   

    def reassign_new_color(self, color_adapt_queue, color_name):
        max_length = color_adapt_queue.max_length
        if (color_adapt_queue.get_queue_length(color_name, "hue") == max_length and
            color_adapt_queue.get_queue_length(color_name, "dev") == max_length and
            color_adapt_queue.get_queue_length(color_name, "sat") == max_length):
            # print(f"COLOR {color_name} was REASSIGNED. NOW CAN BE DETECTED")
            # print(f"---{color_name}---")
            # print(f"HUE QUEUE for {color_name}: {color_adapt_queue.queues[color_name]['hue']}")
            # print(f"DEV QUEUE for {color_name}: {color_adapt_queue.queues[color_name]['dev']}")
            # print(f"SAT QUEUE for {color_name}: {color_adapt_queue.queues[color_name]['sat']}")
            # print(f"----------")
            self.calib_values[color_name]["reassigned"] = True
            # print(color_adapt_queue)
        return

class ColorQueue:
    def __init__(self, color_settings, max_length=4):
        self.max_length = max_length
        self.color_settings = color_settings
        self.queues = {}
        self.initialize_queues()

    def initialize_queues(self):
        """Initializes empty queues for each parameter of each color."""
        for color_name in self.color_settings.colors:
            self.queues[color_name] = {
                "hue": deque(maxlen=self.max_length),
                "dev": deque(maxlen=self.max_length),
                "sat": deque(maxlen=self.max_length)
            }

    def __str__(self):
        output = []
        for color_name, queue_dict in self.queues.items():
            output.append(f"---{color_name}---")
            output.append(f"HUE QUEUE for {color_name}: {queue_dict['hue']}")
            output.append(f"DEV QUEUE for {color_name}: {queue_dict['dev']}")
            output.append(f"SAT QUEUE for {color_name}: {queue_dict['sat']}")
            output.append("----------")
        return "\n".join(output)

    def add(self, color_name, parameter, value):
        """Adds a value to the specified parameter queue of the specified color."""
        if color_name in self.queues and parameter in self.queues[color_name]:
            self.queues[color_name][parameter].append(value)
        else:
            raise ValueError("Invalid color name or parameter")

    def calculate_weights(self, color_name, parameter):
        """Generates weights with a Gaussian-like distribution for the specified parameter queue."""
        queue = self.queues[color_name][parameter]
        num_elements = len(queue)
        if num_elements == 0:
            return []

        x = np.linspace(-3, 3, num_elements)
        weights = norm.pdf(x)
        # Normalize such that the max weight is not less than 0.5
        weights /= weights.max() / 0.5
        return weights

    def weighted_average(self, color_name, parameter):
        """Calculates the weighted average of values in the specified parameter queue."""
        if color_name not in self.queues or parameter not in self.queues[color_name]:
            raise ValueError("Invalid color name or parameter")

        weights = self.calculate_weights(color_name, parameter)
        queue = self.queues[color_name][parameter]

        if len(weights) != len(queue):
            raise ValueError("The number of weights must match the number of values")

        return np.average(queue, weights=weights)
    
    def average(self, color_name, parameter):
        """Calculates the average of values in the specified parameter queue."""
        if color_name not in self.queues or parameter not in self.queues[color_name]:
            raise ValueError("Invalid color name or parameter")

        queue = self.queues[color_name][parameter]
        
        return np.mean(queue)
    
    def get_queue_length(self, color_name, parameter):
        """Returns the number of elements in the queue for a specific color."""
        if color_name not in self.queues or parameter not in self.queues[color_name]:
            raise ValueError("Invalid color name or parameter")
        else:
            return len(self.queues[color_name][parameter])
    
    def adjust_detected_colors(self, rectgs, image, color_name, divider_param=6):
        new_hue_value = None
        new_sat_value = None
        for rectg in rectgs:
            rectg_color = Vision.utils.color_value_to_str(rectg.color)
            if rectg_color == color_name:
                rectg_hue, new_sat_value = self.color_settings.calculate_rectangle_color(rectg, image)
                if rectg_hue is not None:
                    color_deviation = self.color_settings.calib_values[color_name]["dev_avg"]
                    
                    lower_threshold = color_deviation // divider_param
                    
                    hue_dif = np.abs(rectg_hue - self.color_settings.calib_values[color_name]["hue_avg"])
                    if (np.abs(hue_dif) > lower_threshold and
                        np.abs(hue_dif) < color_deviation):
                        old_hue = self.color_settings.calib_values[color_name]["hue_avg"]
                        new_hue_value = rectg_hue

        return new_hue_value, new_sat_value
    
def create_mask(image, color_name, color_params):
    
    black_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    if not color_params[color_name]["reassigned"]:
        return black_mask
    
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
        
    # Hue, Saturation, Value masked: 
    # if color_name == "green":
    # cv2.imshow('Masked sat', masked_sat)
    # cv2.imshow('Masked hue', masked_hue)
    # cv2.imshow('Mask', mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return mask