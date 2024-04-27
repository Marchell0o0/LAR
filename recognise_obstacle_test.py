import cv2 # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from scipy.ndimage import gaussian_filter1d # type: ignore
from scipy.signal import find_peaks # type: ignore

from robolab_turtlebot import Turtlebot, get_time, Rate


# blue_color_range = [80, 150]
# blue_color_range = [50, 100]
# blue_color_deviation = 50
# red_color_range = [0, 50]
# green_color_range = [0, 50]
# saturation_range = [40, 220]

class ColorSettings:
    def __init__(
        self, blue_range,
        blue_deviation,
        red_range,
        red_deviation,
        green_range,
        green_deviation,
        # deviation_range not used
        deviation_range,
        saturation_range
        ):
        self.blue_range = blue_range
        self.blue_deviation = blue_deviation
        self.red_range = red_range
        self.red_deviation = red_deviation
        self.green_range = green_range
        self.green_deviation = green_deviation
        self.deviation_range = deviation_range
        self.saturation_range = saturation_range

    def __str__(self):
        return (f"ColorSettings("
                f"blue_range = {self.blue_range}, "
                f"blue_deviation = {self.blue_deviation}, "
                f"red_range = {self.red_range}, "
                f"red_deviation = {self.red_deviation}, "
                f"green_range = {self.green_range}, "
                f"green_deviation = {self.green_deviation}, "
                f"deviation_range = {self.deviation_range}, "
                f"saturation_range = {self.saturation_range})")

def calculate_hue_average(hist_hue, color_range):
    if color_range[0] <= color_range[1]:
        argmax_hue = np.argmax(hist_hue[color_range[0]:color_range[1]]) + color_range[0]
        argmin_hue = np.argmin(hist_hue[color_range[0]:color_range[1]]) + color_range[0]
    else:
        argmax_hue_1 = np.argmax(hist_hue[color_range[0]:180]) + color_range[0]
        argmax_hue_2 = np.argmax(hist_hue[0:color_range[1]])
        argmax_hue = max(argmax_hue_1, argmax_hue_2)
        argmin_hue_1 = np.argmin(hist_hue[color_range[0]:180]) + color_range[0]
        argmin_hue_2 = np.argmin(hist_hue[0:color_range[1]])
        argmin_hue = max(argmin_hue_1, argmin_hue_2)
        
    
    # Calculate the average of the peak and valley
    average_hue = (argmax_hue + argmin_hue) // 2
    
    return average_hue

def calculate_sat_average(hist_sat, saturation_range):
    # Find the peak hue value within the range
    argmax_saturation = np.argmax(hist_sat[saturation_range[0]:saturation_range[1]]) + saturation_range[0]
    
    # Find the minimum hue value (valley) within the same range
    argmin_saturation = np.argmin(hist_sat[saturation_range[0]:saturation_range[1]]) + saturation_range[0]
    # Calculate the average of the peak and valley
    sat_threshold = (argmax_saturation + argmin_saturation) // 2
    return sat_threshold

def smooth_histogram_with_gaussian(hist, sigma = 1):
    hist_smoothed = gaussian_filter1d(hist.ravel(), sigma = sigma)
    return hist_smoothed

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

def calculate_hue_params(image, color_range, deviation_range):
    
    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate the histogram for the hue channel
    hist_hue = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    
    # hist_hue_smoothed = smooth_histogram(hist_hue, window_size=10)
    hist_hue_smoothed = smooth_histogram_with_gaussian(hist_hue, sigma = 3)
    
    average_hue = calculate_hue_average(hist_hue_smoothed, color_range)
    
    
    # Calculate standard deviation around the peak
    color_deviation = np.std(hsv[:,:,0][(hsv[:,:,0] > average_hue - deviation_range) & (hsv[:,:,0] < average_hue + deviation_range)])
    
    print("Avarage color: ", average_hue)
    print("Color deviation: ", color_deviation)

    return average_hue, color_deviation, hist_hue_smoothed

# use this function for a green obstacle
# def calculate_saturation_threshold(image, saturation_range):
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
#     # Calculate the histogram for the saturation channel
#     hist_sat = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    
#     hist_sat_smoothed = smooth_histogram_with_gaussian(hist_sat, sigma = 3)
    
#     # Invert the histogram to find valleys as peaks
    
#     relevant_histogram = hist_sat_smoothed[saturation_range[0]:saturation_range[1]+1]
    
#     inverted_histogram = -relevant_histogram

#     # Use find_peaks to find the indexes of these peaks (valleys in the original histogram)
#     peaks, _ = find_peaks(inverted_histogram)

#     # plot_histogram(inverted_histogram, "saturation")
    
#     # If no peaks are found, use a default threshold, or handle this case as needed
#     if peaks.size > 0:
#         print("Saturation value: ", peaks[0] + saturation_range[0])
#         # return peaks[0] + saturation_range[0]
#         return 50
#     else:
#         print("Defaut saturation value returned: 100")
#         return 100

# use this function for a blue and red obstacle
def calculate_saturation_threshold(image, saturation_range):
    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate the histogram for the saturation channel
    hist_sat = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    
    hist_sat_smoothed = smooth_histogram_with_gaussian(hist_sat, sigma=3)
    
    # Find an adaptive threshold for saturation
    # Use this threshold for blue and red obstacle
    sat_threshold = np.argmin(hist_sat[saturation_range[0]:saturation_range[1]]) + saturation_range[0]
    # sat_threshold = calculate_sat_average(hist_sat_smoothed, saturation_range)
    
    print("Saturation avarage: ", sat_threshold)
    
    # Debugging:
    # plot_histogram(hist_sat_smoothed, "saturation")
    
    return sat_threshold

def detect_labels(image_mask, image, min_area = 1000, min_aspect_ratio = 2):
    result = np.zeros_like(image_mask)
    original_image = image.copy()
    
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_mask, 4, cv2.CV_32S)
    for label in range(1, numLabels):  # Start from 1 to skip the background
        area = stats[label, cv2.CC_STAT_AREA]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        if height > 0 and width > 0:
            aspect_ratio = float(height/width)
        else:
            aspect_ratio = 0
        if area > min_area and aspect_ratio > min_aspect_ratio:
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            (cX, cY) = centroids[label]
            
            # Debugging:
            # print("-----------------------")
            # print(f"LABEL:")
            # print(f"Area: {area}")
            # print(f"x, y: {x, y}")
            # print(f"Width: {width}")
            # print(f"Height: {height}")
            # print(f"Centroid: {cX, cY}")
            # print("-----------------------")
            # print(" ")
            
            component_mask = (labels == label).astype("uint8") * 255
            # cv2.rectangle(component_mask, (x, y), (x + width, y + height), (0, 255, 0), 3)
            result = cv2.bitwise_or(result, component_mask)
            cv2.rectangle(original_image, (x, y), (x + width, y + height), (255, 255, 255), 2)
            cv2.putText(original_image, "Label", (int(cX - width), int(cY)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # Debugging:
            # cv2.imshow('Debugging image', original_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
    return result, original_image

def detect_rectangles(image_mask, image, epsilon = 0.02, min_area = 1000):
    counter = 0
    result = np.zeros_like(image_mask)
    original_image = image.copy()

    image_mask = cv2.GaussianBlur(image_mask, (7, 7), 0)
    # image_mask = cv2.blur(image_mask,(9,9))

    # Find edges in the image using Canny
    # edges = cv2.Canny(image_mask, 50, 150)

    contours, _ = cv2.findContours(image_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    potential_rectangles = []
    
    if not contours:
        print("No contours found.")
        return result, original_image

    # Sort contours by area and find the largest one
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)
    print("Num of contours found: ", len(contours))
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, peri * epsilon, True)
        area = cv2.contourArea(approx)
        
        M = cv2.moments(cnt)
        # m00 will never be zero for polygons without self-intesections
        if M["m00"] == 0:
            print("Polygon was self-intersected")
            continue
        
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        num_vertices = len(approx)    
        arc_length = cv2.arcLength(approx, True)
        x, y, w, h = cv2.boundingRect(approx)
        min_area_rect = cv2.minAreaRect(approx)
        box = cv2.boxPoints(min_area_rect)
        box = np.int0(box) # Convert to integer
        
        if num_vertices <= 5 and area > min_area:
            counter += 1
            potential_rectangles.append(approx)
            # cv2.polylines(result, [approx], True, (0, 255, 0), 2)
            cv2.drawContours(result, [box], 0, (255, 255, 255), cv2.FILLED)
            cv2.drawContours(original_image, [approx], 0, (0, 255, 0), 2)
            cv2.drawContours(original_image, [box], 0, (0, 255, 255), 2)
            cv2.putText(original_image, "Rectangle", (cX, cY),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            # Debugging
            # cv2.imshow('Debugging image', original_image)
            # cv2.imshow('Debugging masked', result)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # Output the calculated details
            print("-----------------------")
            print("RECTANGLE:")
            print(f"Area: {area}")
            print(f"Arc Length: {arc_length}")
            print(f"Moment: {cX, cY}")
            print(f"Bounding Rect: {x}, {y}, {w}, {h}")
            print(f"Min Area Rect: {min_area_rect}")
            print(f"Box Points: {box}")
            print("-----------------------")
            print(" ")
            # break
        else:
            print("No rectangles found")
            # break

    print("Number of rectangles: ", len(potential_rectangles))
    return result, original_image

def recognise_obstacle(image, color_range, deviation_range, saturation_range):
    # Preprocess the image increasing its contast
    # clahe_model = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1,1))
    # colorimage_b = clahe_model.apply(image[:,:,0])
    # colorimage_g = clahe_model.apply(image[:,:,1])
    # colorimage_r = clahe_model.apply(image[:,:,2])
    # colorimage_clahe = np.stack((colorimage_b,colorimage_g,colorimage_r), axis=2)
    # image = colorimage_clahe
      
    hue_value, hue_deviation, hist_hue_smoothed = calculate_hue_params(
        image, 
        color_range,
        deviation_range
        )
    saturation_threshold = calculate_saturation_threshold(image, saturation_range)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_channel, saturation_channel, value_channel = cv2.split(hsv)
    
    masked_sat = np.where((saturation_channel > saturation_threshold), 255, 0).astype(np.uint8)
    masked_hue = np.where((np.abs(hue_channel - hue_value) < hue_deviation), 255, 0).astype(np.uint8)
    masked = np.where(((masked_hue == 255) & (masked_sat == 255)), 255, 0).astype(np.uint8)
    
    masked_labels, image_labels = detect_labels(masked, image)
    masked_rectangles, image_rectangles = detect_rectangles(masked_labels, image_labels)
    
    # Debugging:
    # plot_histogram(hist_hue_smoothed, "hue")
    
    # Uncomment to see what you want:
    
    #Original:
    # cv2.imshow('Original', image)
    # cv2.imshow('Contrast original', colorimage_clahe)
    # cv2.imshow('Hue', hue_visualized)
    # cv2.imshow('Saturation', saturation_channel)
    # cv2.imshow('Value', value_channel)
    
    # Hue, Saturation, Value masked: 
    cv2.imshow('Masked sat', masked_sat)
    cv2.imshow('Masked hue', masked_hue)
    # cv2.imshow('Masked', masked)

    # Result: 
    cv2.imshow('Masked labels', masked_labels)
    # cv2.imshow('Labels on the image', image_labels)
    cv2.imshow('Masked rectangles', masked_rectangles)
    cv2.imshow('Rectangles on the image', image_rectangles)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_pc(pc_image):
    x_range = (-1, 1)
    z_range = (0.01, 3.0)
    
    # mask out floor points
    mask_pc = pc_image[:, :, 1] > x_range[0]

    # mask point too far and close
    mask_pc = np.logical_and(mask_pc, pc_image[:, :, 2] > z_range[0])
    mask_pc = np.logical_and(mask_pc, pc_image[:, :, 2] < z_range[1])

    # empty image
    image_result = np.zeros(mask_pc.shape)

    # assign depth i.e. distance to image
    image_result[mask_pc] = np.int8(pc_image[:, :, 2][mask_pc] / 3.0 * 255)
    im_color = cv2.applyColorMap(255 - image_result.astype(np.uint8),
                                    cv2.COLORMAP_JET)
    
    # cv2.imshow("Point cloud", im_color)

# Doesnt work yet
def test_depth(depth_image):
    mask_depth = depth_image[:, :] 
    masked_depth = np.where((depth_image[:, :] < 1.1 and depth_image[:, :] > 0.9), 255, 0).astype(np.uint8)
    mask_depth = np.logical_and(mask_depth, depth_image[:, :] < 1.1)
    mask_depth = np.logical_and(mask_depth, depth_image[:, :] > 0.9)
    
    # cv2.imshow("Depth", masked_depth)
    # x_depth_point = 
    # y_depth_point = 
    # print(f"Depth to the 643, 306: {}, {}")
    
def main():
    color_settings = ColorSettings(
        blue_range = [70, 130],
        blue_deviation = 50,
        red_range = [0, 7],
        red_deviation = 50,
        green_range = [20, 60],
        green_deviation = 50,
        # deviation_range not used
        deviation_range = 10,
        saturation_range = [20, 235]
    ) 
    
    turtle = Turtlebot(rgb = True)

    turtle.wait_for_rgb_image()

    image = turtle.get_rgb_image()

    # Test Point Cloud:
    # pc_image = np.load("test_images/spins/meas03/x01_1_pc.npy")
    # test_pc(pc_image)
    
    # Test Depth image:
    # depth_image = np.load("test_images/spins/meas01/x01_1_depth.npy")
    # test_depth(depth_image)
    
    # Test obstacle recognition:
    # One photo:
    # image_name = "test_images/spins/meas01/x01_1_image.npy"
    # image = np.load(image_name)
    recognise_obstacle(
            image,
            color_settings.blue_range,
            color_settings.blue_deviation,
            color_settings.saturation_range
    )
    
    # All photos:
    # for i in range(1, 14):
    #     # image = np.load('test_images/images/x01_image.npy')
    #     image_name = "test_images/spins/meas01/x01_{}_image.npy".format(i)
    #     image = np.load(image_name)
    #     print(f"File no. {i}")
    #     recognise_obstacle(
    #         image,
    #         color_settings.red_range,
    #         color_settings.red_deviation,
    #         color_settings.saturation_range
    #         )


if __name__ == "__main__":
    main()