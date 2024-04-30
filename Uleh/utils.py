import cv2 # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from scipy.ndimage import gaussian_filter1d # type: ignore
import math

def morphology_preprocessing(masked_hue):

    # Create a rectangular kernel for morphological operations
    kernel_size = 2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Apply morphological opening to remove small white noise
    opened_mask = cv2.morphologyEx(masked_hue, cv2.MORPH_OPEN, kernel, iterations=1)

    # Optionally apply morphological closing to close small holes within the objects
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return opened_mask, closed_mask

def color_mask(hsv_image, lowerLimit, upperLimit):
    mask = cv2.inRange(hsv_image, lowerLimit, upperLimit)
    return mask

def get_limits(color):
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  # Get the hue value

    # Handle red hue wrap-around
    if hue >= 165:  # Upper limit for divided red hue
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:  # Lower limit for divided red hue
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit

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

# Preprocess the image increasing its contast
def clahe_preprocess(image):
    clahe_model = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1,1))
    colorimage_b = clahe_model.apply(image[:,:,0])
    colorimage_g = clahe_model.apply(image[:,:,1])
    colorimage_r = clahe_model.apply(image[:,:,2])
    colorimage_clahe = np.stack((colorimage_b,colorimage_g,colorimage_r), axis=2)
    
    return colorimage_clahe
    
def is_within_range(value, range_bounds):
    return range_bounds[0] <= value <= range_bounds[1]

def draw_rectangles(result_mask, original_image, rectangle):
    rectangle_color = (int(rectangle.color), int(rectangle.color), int(rectangle.color))
    # cv2.polylines(result_mask, [rectangle.approx], True, (0, 255, 0), 2)
    cv2.drawContours(result_mask, [rectangle.box_points], 0, rectangle_color, cv2.FILLED)
    cv2.drawContours(original_image, [rectangle.approx], 0, (0, 255, 255), 2)
    cv2.putText(original_image,
                f"Z1: {rectangle.distance:.2f} m",
                # (int(rectangle.cX - rectangle.width * 0.5),
                #  int(rectangle.cY - rectangle.height * 0.6)),
                (int(rectangle.cX),
                 int(rectangle.cY)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(original_image,
                f"Z2: {rectangle.distance2:.2f} m",
                (int(rectangle.cX),
                 int(rectangle.cY - rectangle.height * 0.3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(original_image,
                f"Z3: {rectangle.distance3:.2f} m",
                (int(rectangle.cX),
                 int(rectangle.cY + rectangle.height * 0.3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(original_image,
                f"A: {rectangle.angle_pos:.2f} deg",
                (int(rectangle.cX),
                 int(rectangle.cY + rectangle.height * 0.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 0, 255), 2)
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
    cv2.drawContours(original_image, [rectangle.box_points], 0, draw_color, 2)
    cv2.putText(original_image, rectg_name, (rectangle.cX, rectangle.cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)
    points = calculate_rectangle_points(rectangle)
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

    # Display using matplotlib
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
    """
    Calculate the angle in radians and distance from the camera to a point.
    
    Parameters:
    y (float): horizontal displacement from the center (in meters)
    z (float): distance from the camera to the object (in meters)

    Returns:
    (float, float): tuple of (angle in radians, distance)
    """
    if z == 0 and z != None and y != None:
        raise ValueError("Distance z cannot be zero.")

    angle = np.arcsin(y / z)  # Result is in radians
    
    # Convert to degrees for printing out
    angle_deg = (360 * angle) / (2* math.pi)
    
    print("Angle (radians):", angle)
    print("Angle (degrees):", int(angle_deg))

    return angle, angle_deg

def calculate_rectangle_points(rectangle):
    cX, cY = rectangle.cX, rectangle.cY
    width, height = rectangle.width, rectangle.height
    # TODO: find out how to do it smarter
    if rectangle.angle_rot > 45:
        rectangle.angle_rot -= 90 
        print(f"New angle_rot is {rectangle.angle_rot}")
    angle_rad = math.radians(rectangle.angle_rot)  # Convert angle from degrees to radians

    # Function to rotate a point around center by angle
    def rotate_point(x, y, cx, cy, angle):
        cos_theta, sin_theta = math.cos(angle), math.sin(angle)
        x, y = x - cx, y - cy
        nx = x * cos_theta - y * sin_theta + cx
        ny = x * sin_theta + y * cos_theta + cy
        return int(nx), int(ny)
    
    # Center of the rectangle
    center = (cX, cY)

    # Corners without rotation
    lt = (cX - width / 3, cY - height / 3)
    rt = (cX + width / 3, cY - height / 3)
    lb = (cX - width / 3, cY + height / 3)
    rb = (cX + width / 3, cY + height / 3)

    # Midpoints without rotation
    top = (cX, cY - height / 3)
    bottom = (cX, cY + height / 3)
    left = (cX - width / 3, cY)
    right = (cX + width / 3, cY)

    # Rotate points
    points = [lt, rt, lb, rb, top, bottom, left, right]
    rotated_points = [rotate_point(px, py, cX, cY, angle_rad) for (px, py) in points]

    # Assign rotated points back to meaningful variable names
    (lt, rt, lb, rb, top, bottom, left, right) = rotated_points

    # return {
    #     "center": center,
    #     "corners": {"lt": lt, "rt": rt, "lb": lb, "rb": rb},
    #     "midpoints": {"top": top, "bottom": bottom, "left": left, "right": right}
    # }
    return [center, top, bottom, left, right, lt, rt, lb, rb]
    
def draw_points_on_image(image, points):
    color = (0, 255, 0)
    radius = 3  # Small radius for a dot-like appearance
    thickness = 1  # Fill the circle
    
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
    