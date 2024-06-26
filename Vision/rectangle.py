import cv2
import numpy as np
from typing import List, Type

import Vision.utils
import Vision.color
import Vision.depth

class Rectangle:
    def __init__(self,
                 area,
                 centerX,
                 centerY,
                 width,
                 height,
                 angle_rot,
                 box_points,
                 major_points,
                 approx,
                 aspect_ratio,
                 color,
                 distance,
                 angle_pos,
                 rectangle_y):
        self.area = area
        self.cX = centerX
        self.cY = centerY
        self.width = width
        self.height = height
        self.angle_rot = angle_rot
        self.box_points = box_points
        self.major_points = major_points
        self.approx = approx
        self.aspect_ratio = aspect_ratio
        self.color = color
        self.distance = distance
        self.angle_pos = angle_pos
        self.y = rectangle_y
    
    def __str__(self):
        return (f"-----------------------\n"
                "RECTANGLE:\n"
                f"Area: {self.area}\n"
                f"Moment: ({self.cX}, {self.cY})\n"
                f"Width and Height: {self.width}, {self.height}\n"
                f"Angle_rot: {self.angle_rot}\n"
                f"Box Points: {self.box_points}\n"
                f"Aspect ratio: {self.aspect_ratio}\n"
                f"Color: {self.color}\n"
                f"Distance: {self.distance}\n"
                f"Angle: {self.angle_pos}\n"
                "-----------------------\n ")

class RectangleProcessor:
    def __init__(self,
                 image,
                 pc_image,
                 color_settings,
                 color_adapt_queue,
                 show_image=False) -> None:
        
        self.image = image
        self.pc_image = pc_image
        self.color_settings = color_settings
        self.color_adapt_queue = color_adapt_queue
        self.rectangles: List[Rectangle] = []
        self.show_image = show_image

    def detect_labels(self, 
                    image_mask,
                    min_area = 800,
                    aspect_ratio_range = [2, 8]
                    ):
        
        result_mask = np.zeros_like(image_mask)
        
        numLabels, labels, stats, _ = cv2.connectedComponentsWithStats(
            image_mask,
            4,
            cv2.CV_32S
            )
        for label in range(1, numLabels):  # Start from 1 to skip the background
            area = stats[label, cv2.CC_STAT_AREA]
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            if height > 0 and width > 0:
                aspect_ratio = float(height/width)
            else:
                aspect_ratio = 0
            if area > min_area and Vision.utils.is_within_range(aspect_ratio,
                                                        aspect_ratio_range):
                x = stats[label, cv2.CC_STAT_LEFT]
                y = stats[label, cv2.CC_STAT_TOP]
                
                # Get the most common non-zero value within the label's bound box
                
                component_mask = ((labels == label).astype("uint8") 
                                * 255)
                result_mask = np.maximum(result_mask, component_mask)
                cv2.rectangle(result_mask,
                              (x, y),
                              (x + width, y + height),
                              (0, 255, 0),
                              3
                              )
                
        return result_mask

    def detect_rectangles(self, 
                        masked_labels,
                        original_image,
                        color_name,
                        epsilon = 0.02,
                        vertices_range = [4, 6],
                        min_area = 800,
                        aspect_ratio_range = [2, 8],
                        outlier_distance=0.1,
                        cylinder_rad = 0.025):
            
        result_mask = np.zeros_like(masked_labels)

        image_mask = cv2.GaussianBlur(masked_labels, (7, 7), 0)

        contours, _ = cv2.findContours(image_mask,
                                    cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE
                                    )
        
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, peri * epsilon, True)
            area = cv2.contourArea(approx)
            num_vertices = len(approx)    
            x, y, width, height = cv2.boundingRect(approx)
            if height > 0 and width > 0:
                aspect_ratio = float(height/width)
            else:
                aspect_ratio = 0
            
            M = cv2.moments(cnt)
            # m00 will never be zero for polygons without self-intesections
            if M["m00"] == 0:
                continue
            
            if (area > min_area and
                Vision.utils.is_within_range(aspect_ratio, aspect_ratio_range)):
                if Vision.utils.is_within_range(num_vertices, vertices_range):
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    min_area_rect = cv2.minAreaRect(approx)
                    angle_rot = min_area_rect[2]
                    box_points = cv2.boxPoints(min_area_rect)
                    box_points = np.int0(box_points) # Convert to integer
                    
                    label_color_value = Vision.utils.str_to_color_value(color_name)
                    
                    y_coords = []
                    x_coords = []
                    distances = []
                    true_distance = None
                    
                    points = Vision.utils.calculate_rectangle_points(self.image, cX, cY, height, width, angle_rot)
                    for point in points[:3]:
                        rectangle_x, rectangle_y, rectangle_distance = Vision.depth.find_point_pc_coords(self.pc_image,
                                                                    point[0],
                                                                    point[1]
                                                                    )
                        if (not np.isnan(rectangle_y) and
                            rectangle_y is not None and
                            not np.isnan(rectangle_distance) and
                            rectangle_distance is not None and
                            not np.isnan(rectangle_x) and
                            rectangle_x is not None):
                            
                            true_distance = np.sqrt(rectangle_distance**2 - np.abs(rectangle_x)**2) + cylinder_rad
                            
                            y_coords.append(rectangle_y)
                            distances.append(true_distance)
                            x_coords.append(rectangle_x)
                    
                    Vision.utils.remove_values_excluding_outliers(distances, outlier_distance) 
                    if len(distances) != 0 and len(y_coords) != 0: 
                        rectangle_distance = np.mean(distances)   
                        rectangle_y = np.mean(y_coords)

                        rectangle_angle = Vision.utils.calculate_angle(rectangle_y, rectangle_distance)
                        
                        if (rectangle_angle is not None and
                            Vision.utils.is_within_range_distance(aspect_ratio, rectangle_distance, area)):
                    
                            rectangle = Rectangle(
                                                area,
                                                cX,
                                                cY,
                                                width,
                                                height,
                                                angle_rot,
                                                box_points,
                                                points,
                                                approx,
                                                aspect_ratio,
                                                label_color_value,
                                                rectangle_distance,
                                                rectangle_angle,
                                                rectangle_y)
                            
                            self.rectangles.append(rectangle)
                            Vision.utils.draw_rectangle(result_mask, original_image, rectangle)
        
        return result_mask, original_image
    

    def process_image(self):
        colors = ["blue", "green", "red"]
        result_masked = {}
        green_mask = None
        
        if self.image is not None and self.pc_image is not None:
            
            original_image = self.image.copy()
            
            for color_name in colors:
                
                result_masked[color_name] = Vision.color.create_mask(self.image,
                                                            color_name,
                                                            self.color_settings.calib_values)
            
                masked_labels = self.detect_labels(result_masked[color_name])
                masked_rectangles, image_rectangles = self.detect_rectangles(masked_labels, original_image, color_name)
                if color_name == "green":
                    green_mask = result_masked[color_name]
            
            if green_mask is not None:
                masked_labels = green_mask
                  
            self.color_settings.update_image_colors(
                self.rectangles,
                self.image,
                self.color_adapt_queue)
            
            cylinders = []
            
            for rectg in self.rectangles:
                # R
                if rectg.color == 250:
                    rectg_color = 0
                # B
                elif rectg.color == 150:
                    rectg_color = 1
                # G
                elif rectg.color == 200:
                    rectg_color = 2
                # Unknown color    
                else:
                    rectg_color = 3
                    
                if rectg_color != 3:
                    cylinders.append((rectg.distance, -rectg.angle_pos, rectg_color))
            
            if self.show_image == True:
                # cv2.imshow('Masked labels', masked_labels)
                # cv2.imshow('Labels on the image', image_labels)
                # cv2.imshow('Masked rectangles', masked_rectangles)
                cv2.imshow('Masked labels', masked_labels)
                cv2.imshow('Rectangles on the image', image_rectangles)
                # depth.generate_pc_image(pc_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            cylinders = None
            masked_rectangles = None
            image_rectangles = None
            self.rectangles = None

        return cylinders, masked_labels, image_rectangles, self.rectangles