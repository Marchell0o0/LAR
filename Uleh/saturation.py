import cv2 # type: ignore
import numpy as np # type: ignore
# from scipy.signal import find_peaks # type: ignore

from Uleh.utils import smooth_histogram_with_gaussian

# use this function for a blue and red obstacle
def calculate_saturation_threshold(image, saturation_range, color_name):
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate the histogram for the saturation channel
    hist_sat = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    
    hist_sat_smoothed = smooth_histogram_with_gaussian(hist_sat, sigma=3)
    
    # Find an adaptive threshold for saturation
    # Use this threshold for blue and red obstacle
    sat_threshold = np.argmin(hist_sat[saturation_range[0]:saturation_range[1]]) + saturation_range[0]
    # sat_threshold = calculate_sat_average(hist_sat_smoothed, saturation_range)
      
    # Debugging:
    # print(f"Saturation avarage for {color_name}: {sat_threshold}")
    # plot_histogram(hist_sat_smoothed, "saturation")
    
    return sat_threshold

# use this function for a green obstacle
def calculate_saturation_threshold_green(image, saturation_range):
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate the histogram for the saturation channel
    hist_sat = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    
    hist_sat_smoothed = smooth_histogram_with_gaussian(hist_sat, sigma = 3)
    
    # Invert the histogram to find valleys as peaks
    relevant_histogram = hist_sat_smoothed[saturation_range[0]:saturation_range[1]+1]
    inverted_histogram = -relevant_histogram

    # Use find_peaks to find the indexes of these peaks (valleys in the original histogram)
    
    # peaks, _ = find_peaks(inverted_histogram)

    # # plot_histogram(inverted_histogram, "saturation")
    
    # if peaks.size > 0:
    #     #TODO: remove hardcoding saturation for green
    #     # print("Saturation value for green: ", peaks[0] + saturation_range[0])
    #     # return peaks[0] + saturation_range[0]
        
    #     # print("Saturation value for green: ", 70)
    #     return 70
    # else:
    #     # print("Defaut saturation value returned: 40")
    #     return 70
    
    return 70