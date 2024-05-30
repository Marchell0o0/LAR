import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import math

def color_mask(hsv_image, lowerLimit, upperLimit):
    mask = cv2.inRange(hsv_image, lowerLimit, upperLimit)
    return mask

def plot_histogram(hist, type):
    plt.figure()
    if type == "saturation":
        plt.title('Saturation Histogram')
        plt.xlabel('Saturation')
        plt.xlim([0, 255])
    elif type == "hue":
        plt.title('Hue Histogram')
        plt.xlabel('Hue')
        plt.xlim([0, 180])
    plt.ylabel('Frequency')
    plt.plot(hist)
    plt.show()
    
def smooth_histogram_with_gaussian(hist, sigma = 1):
    hist_smoothed = gaussian_filter1d(hist.ravel(), sigma = sigma)
    return hist_smoothed
    
def is_within_range(value, range_bounds):
    return range_bounds[0] <= value <= range_bounds[1]

def is_within_range_distance(aspect_ratio, distance, area):
    is_in_range = False
    if (is_within_range(distance, [0.15, 0.5]) and
        is_within_range(aspect_ratio, [2, 4]) and
        is_within_range(area, [18000, 70000])):
        is_in_range = True
    elif (is_within_range(distance, [0.3, 1]) and
        is_within_range(aspect_ratio, [3, 6.2]) and
        is_within_range(area, [5000, 60000])):
        is_in_range = True
    elif (is_within_range(distance, [1, 1.5]) and
        is_within_range(aspect_ratio, [3.1, 7]) and
        is_within_range(area, [2200, 10000])):
        is_in_range = True
    elif (is_within_range(distance, [1.3, 2]) and
        is_within_range(aspect_ratio, [3.1, 7.8]) and
        is_within_range(area, [800, 5000])):
        is_in_range = True
    # elif (is_within_range(distance, [2, 3]) and
    #     is_within_range(aspect_ratio, [3.7, 8]) and
    #     is_within_range(area, [800, 2000])):
    #     is_in_range = True
    return is_in_range
        
        
def draw_rectangle(result_mask, original_image, rectangle):
    cX, cY = rectangle.cX, rectangle.cY
    height, width = rectangle.height, rectangle.width
    box_points = rectangle.box_points
    approx = rectangle.approx
    distance = rectangle.distance
    points = rectangle.major_points
    angle_pos = rectangle.angle_pos
    y = rectangle.y
    color_bgr = (int(rectangle.color), int(rectangle.color), int(rectangle.color))
    rectg_name = "None"
    draw_color = (0, 0, 0)
    if rectangle.color == 150:
        rectg_name = "Blue"
        draw_color = (255, 0, 0)
    elif rectangle.color == 200:
        rectg_name = "Green"
        draw_color = (0, 255, 0)
    elif rectangle.color == 250:
        rectg_name = "Red"
        draw_color = (0, 0, 255)
    if rectg_name == "None":
        print(f"Nothing can be drawn with a color value: {rectangle.color}")
        return
    cv2.drawContours(result_mask, [box_points], 0, color_bgr, cv2.FILLED)
    cv2.drawContours(original_image, [approx], 0, (0, 255, 255), 2)
    cv2.drawContours(original_image, [box_points], 0, draw_color, 2)
    cv2.putText(original_image,
                f"Z: {distance:.2f} m",
                (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(original_image,
                f"Y: {y:.2f} m",
                (cX, int(cY -height / 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(original_image,
                f"A: {np.degrees(angle_pos):.2f} deg",
                (cX, int(cY + height * 0.3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 0, 255), 2)
    draw_points_on_image(original_image, points)
    return

def generate_HSV_image(image):
    # Convert the image from BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Separate the H, S, and V channels
    H, S, _ = cv2.split(hsv)

    # Calculate the 2D histogram for the hue and saturation
    hist, xbins, ybins = np.histogram2d(H.flatten(), S.flatten(), bins=[180, 256], range=[[0, 180], [0, 256]])

    # Apply logarithmic scaling to the histogram values
    hist_log = np.log1p(hist)

    # Normalize the logarithmically scaled histogram to the range 0 to 255
    hist_norm = cv2.normalize(hist_log, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Generate an HSV image with all possible hue and saturation values with max brightness
    hsv_base = np.zeros((256, 180, 3), dtype=np.uint8)
    hsv_base[..., 0] = np.arange(180)  # Hue
    hsv_base[..., 1] = 255  # Saturation set to the max
    hsv_base[..., 2] = 255  # Value set to the max initially

    # Apply the normalized histogram as a mask for the value channel
    # Rotate the histogram to align with the base HSV image's axes
    hist_norm_rotated = np.rot90(hist_norm)
    hist_norm_flipped = np.flipud(hist_norm_rotated)
    hsv_base[..., 2] = hist_norm_flipped.astype(np.uint8)  # Apply mask

    # Convert the HSV image to RGB for displaying
    rgb_image = cv2.cvtColor(hsv_base, cv2.COLOR_HSV2RGB)

    plt.imshow(rgb_image, origin='lower', extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()])
    plt.title('HSV Color Space with Logarithmic Frequency Brightness')
    plt.xlabel('Hue')
    plt.ylabel('Saturation')
    plt.show()
    
def generate_general_HSV():
    # Generate an image with all possible Hue and Saturation values for maximum Value (V = 255)
    hue_range = np.arange(180, dtype=np.uint8)  # Hue values range from 0 to 179
    sat_range = np.arange(256, dtype=np.uint8)  # Saturation values range from 0 to 255
    H, S = np.meshgrid(hue_range, sat_range)
    V = 255 * np.ones_like(H, dtype=np.uint8)  # Max value for V channel

    # Stack to make HSV image
    HSV = np.stack((H, S, V), axis=-1)

    # Convert HSV to RGB for displaying
    RGB = cv2.cvtColor(HSV, cv2.COLOR_HSV2RGB)

    # Display using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(RGB, origin='lower', aspect='auto')
    plt.title('Hue-Saturation Graph with V=255')
    plt.xlabel('Hue')
    plt.ylabel('Saturation')
    plt.xlim([0, 179])
    plt.ylim([0, 255])
    plt.show()
    
def calculate_angle(y, z):    
    if np.isnan(y) or np.isnan(z) or z == 0:
        return None

    ratio = y / z
    
    # Valid range for arcsin
    if ratio < -1 or ratio > 1:        
        return None

    angle = np.arcsin(ratio)
    
    if not isinstance(angle, float):
        return None
    
    # Result is in radians
    return angle

def calculate_rectangle_points(image, cX, cY, height, width, angle_rot, epsilonX=4, epsilonY=3):
    if angle_rot < -45:
        angle_rot = -(angle_rot + 90) 
    elif angle_rot >= 45 and angle_rot <= 90:
        angle_rot -= 90
    angle_rot_rad = math.radians(angle_rot)  # Convert angle from degrees to radians

    def rotate_point(x, y, cx, cy, angle):
        cos_theta, sin_theta = math.cos(angle), math.sin(angle)
        x, y = x - cx, y - cy
        nx = x * cos_theta - y * sin_theta + cx
        ny = x * sin_theta + y * cos_theta + cy
        return int(nx), int(ny)
    
    def adjust_points_to_image_bounds(points, image):
        # Check image resolution
        image_size = image.shape
        IMAGE_HEIGHT, IMAGE_WIDTH = image_size[0], image_size[1]
        
        adjusted_points = []
        for x, y in points:
            x = max(0, min(IMAGE_WIDTH - 1, x))
            y = max(0, min(IMAGE_HEIGHT - 1, y))
            adjusted_points.append((x, y))
            
        return adjusted_points
    
    # Center of the rectangle
    center = (cX, cY)

    # Corners without rotation
    lt = (cX - width / epsilonX, cY - height / epsilonY)
    rt = (cX + width / epsilonX, cY - height / epsilonY)
    lb = (cX - width / epsilonX, cY + height / epsilonY)
    rb = (cX + width / epsilonX, cY + height / epsilonY)

    # Midpoints without rotation
    top = (cX, cY - height / epsilonY)
    bottom = (cX, cY + height / epsilonY)
    left = (cX - width / epsilonX, cY)
    right = (cX + width / epsilonX, cY)

    # Rotate points
    points = [lt, rt, lb, rb, top, bottom, left, right]
    rotated_points = [rotate_point(px, py, cX, cY, angle_rot_rad) for (px, py) in points]
         
    adjusted_points = adjust_points_to_image_bounds(rotated_points, image)        

    # Assign rotated points back to meaningful variable names
    (lt, rt, lb, rb, top, bottom, left, right) = adjusted_points
    
    return [center, top, bottom, left, right, lt, rt, lb, rb]

def remove_values_excluding_outliers(values, threshold):
    if len(values) == 3:
        median_value = np.median(values)
        for value in values:
            if abs(value - median_value) > threshold:
                values.remove(value)
    elif len(values) == 2:
        if max(values) - min(values) > threshold:
            values = []
    else:
        value = []
    return
  
def draw_points_on_image(image, points):
    color = (0, 255, 0)
    radius = 3  # Small radius for a dot-like appearance
    thickness = 1
    
    for point in points:
        cv2.circle(image, point, radius, color, thickness)
        
    return

def color_value_to_str(color_value):
    if color_value == 150:
        return "blue"
    elif color_value == 200:
        return "green"
    if color_value == 250:
        return "red"
    else:
        return "None"
    
def str_to_color_value(str):
    if str == "blue":
        return 150
    elif str == "green":
        return 200
    if str == "red":
        return 250
    else:
        return 0
    