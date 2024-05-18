import cv2 # type: ignore
import numpy as np # type: ignore
from typing import List, Type

import Uleh.utils
import Uleh.color
import Uleh.depth

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
                # f"Major Points: {self.major_points}\n"
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

    def detect_labels(self, image_mask,
                    min_area = 800,
                    aspect_ratio_range = [2, 8]
                    ):
        
        result_mask = np.zeros_like(image_mask)
        original_image = self.image.copy()
        
        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
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
            if area > min_area and Uleh.utils.is_within_range(aspect_ratio,
                                                        aspect_ratio_range):
                x = stats[label, cv2.CC_STAT_LEFT]
                y = stats[label, cv2.CC_STAT_TOP]
                
                # Get the most common non-zero value within the label's bound box
                label_color_value = np.bincount(
                    image_mask[y:y+height, x:x+width].flatten()).argmax()
                
                component_mask = ((labels == label).astype("uint8") 
                                * label_color_value)
                result_mask = np.maximum(result_mask, component_mask)
                cv2.rectangle(result_mask,
                              (x, y),
                              (x + width, y + height),
                              (0, 255, 0),
                              3
                              )
                
                # cv2.rectangle(original_image,
                #             (int(x), int(y)),
                #             (int(x + width), int(y + height)),
                #             (255, 255, 255),
                #             2
                #             )
                # cv2.putText(original_image,
                #             "Label",
                #             (int(cX - width),
                #             int(cY)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return result_mask, original_image

    def detect_rectangles(self, 
                        image_mask,
                        image_labels,
                        epsilon = 0.02,
                        vertices_range = [4, 6],
                        min_area = 800,
                        aspect_ratio_range = [2, 8],
                        outlier_distance=0.1,
                        cylinder_rad = 0.025):
        result_mask = np.zeros_like(image_mask)
        original_image = image_labels.copy()

        image_mask = cv2.GaussianBlur(image_mask, (7, 7), 0)
        # cv2.imshow('Blured mask', image_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # TODO: remove almost_rectg_counter
        # almost_rectg_counter = 0
        
        pc_error_counter = 0

        # image_mask = cv2.blur(image_mask,(7,7))

        # Find edges in the image using Cannyq
        # edges = cv2.Canny(image_mask, 50, 150)

        contours, _ = cv2.findContours(image_mask,
                                    cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE
                                    )
        
        if not contours:
            # print("No contours found.")
            return self.rectangles, result_mask, original_image

        # print("Num of contours found: ", len(contours))
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
                # print("Polygon was self-intersected")
                continue
            
            if (area > min_area and
                Uleh.utils.is_within_range(aspect_ratio, aspect_ratio_range)):
                # TODO: remove almost_rectg_counter
                # almost_rectg_counter += 1
                if Uleh.utils.is_within_range(num_vertices, vertices_range):
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    min_area_rect = cv2.minAreaRect(approx)
                    angle_rot = min_area_rect[2]
                    box_points = cv2.boxPoints(min_area_rect)
                    box_points = np.int0(box_points) # Convert to integer
                    
                    # Get the most common non-zero value within the label's bound box
                    label_color_value = np.bincount(
                        image_mask[y:y+height, x:x+width].flatten()).argmax()
                    
                    y_coords = []
                    x_coords = []
                    distances = []
                    true_distance = None
                    
                    points = Uleh.utils.calculate_rectangle_points(self.image, cX, cY, height, width, angle_rot)
                    for point in points[:3]:
                        rectangle_x, rectangle_y, rectangle_distance = Uleh.depth.find_point_pc_coords(self.pc_image,
                                                                    point[0],
                                                                    point[1]
                                                                    )
                        # TODO: rewrite error checking
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
                    
                    Uleh.utils.remove_values_excluding_outliers(distances, outlier_distance) 
                    if len(distances) != 0 and len(y_coords) != 0: 
                        rectangle_distance = np.mean(distances)   
                        rectangle_y = np.mean(y_coords)
                    # else:
                    #     # print("FINAL Y or DISTANCE is EMPTY")
                    #     pc_error_counter += 1
                        
                        rectangle_angle = Uleh.utils.calculate_angle(rectangle_y, rectangle_distance)
                        
                        if (rectangle_angle is not None and
                            Uleh.utils.is_within_range_distance(aspect_ratio, rectangle_distance, area)):
                    
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
                            Uleh.utils.draw_rectangle(result_mask, original_image, rectangle)
                            # print(rectangle)
                    else:
                        print("FINAL Y or DISTANCE is NaN")
                        pc_error_counter += 1
        
        # print(f"ALMOST RECTANGLES: {almost_rectg_counter}")        
        # print("Number of rectangles: ", len(self.rectangles))
        # print(f"Number of errors for PC: ", pc_error_counter)
        
        return self.rectangles, result_mask, original_image
    

    def process_image(self):
        colors = ["blue", "green", "red"]
        result_masked = {}
        
        if self.image is not None and self.pc_image is not None:
            for color_name in colors:
                
                # TODO: remove color_params argument from create_mask
                result_masked[color_name] = Uleh.color.create_mask(self.image,
                                                            color_name,
                                                            self.color_settings.calib_values)
            
            combined_mask = Uleh.color.merge_masks(result_masked[colors[0]],
                                        result_masked[colors[1]],
                                        result_masked[colors[2]]
                                        )
            
            masked_labels, image_labels = self.detect_labels(combined_mask)
            self.rectangles, masked_rectangles, image_rectangles = self.detect_rectangles(masked_labels, image_labels)
            
            Uleh.color.update_image_colors(
                self.rectangles,
                self.image,
                self.color_settings,
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
        # TODO: WARNING uncomment the first line instead of the second
        # return cylinders, masked_rectangles, image_rectangles, self.rectangles
        return cylinders, combined_mask, image_rectangles, self.rectangles