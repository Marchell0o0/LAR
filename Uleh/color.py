import cv2 # type: ignore
import numpy as np # type: ignore
# import matplotlib.pyplot as plt # type: ignore
from typing import List

import utils
from hue import calculate_hue_params
from saturation import calculate_saturation_threshold, calculate_saturation_threshold_green

class ColorSettings:
    def __init__(
        self, 
        blue_range=[70, 130],
        blue_deviation=50,
        green_range=[35, 65],
        green_deviation=70,
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
        
    def calibrate_color(self, images):
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
    
    def assign_detected_color(self, rectg, image):
        points = utils.calculate_rectangle_points(rectg)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_channel, saturation_channel, value_channel = cv2.split(hsv)
        
        color_name = utils.color_value_to_str(rectg.color)
        # print(f"Hue value in center is: {hue_channel[points[7][1], points[7][0]]}")
        hue_values_new = []
        for (px, py) in points[:3]:
            if np.abs(hue_channel[py, px] - self.calib_values[color_name]["hue_avg"]) < self.calib_values[color_name]["dev_avg"]:
               print(np.abs(hue_channel[py, px] - self.calib_values[color_name]["hue_avg"]))
               hue_values_new.append(hue_channel[py, px]) 
        if len(hue_values_new) != 0:
            print(hue_values_new)
            self.calib_values[color_name]["hue_avg"] = int(np.mean(hue_values_new))
            self.calib_values[color_name]["color_reassigned"] = True
        
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
    # Uncomment to see what you want:
    
    #Original:
    # cv2.imshow('Original', image)
    # cv2.imshow('Contrast original', colorimage_clahe)
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
    combined_mask[(blue_mask == 255) & (combined_mask == 0)] = utils.str_to_color_value("blue")
    combined_mask[(green_mask == 255) & (combined_mask == 0)] = utils.str_to_color_value("green")
    combined_mask[red_mask == 255] = utils.str_to_color_value("red")
    return combined_mask
