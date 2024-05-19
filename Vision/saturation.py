import cv2
import numpy as np
from scipy.signal import find_peaks

from Vision.utils import smooth_histogram_with_gaussian

# use this function for a blue and red obstacle
def calculate_saturation_threshold(image, saturation_range, color_name, sat_offset=0):
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate the histogram for the saturation channel
    hist_sat = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    
    hist_sat_smoothed = smooth_histogram_with_gaussian(hist_sat, sigma=3)
    
    # Find an adaptive threshold for saturation
    sat_threshold = np.argmin(hist_sat_smoothed[saturation_range[0]:saturation_range[1]]) + saturation_range[0]

    return sat_threshold

# use this function for a green obstacle
def calculate_saturation_threshold_green(image, saturation_range, sat_offset=20):
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate the histogram for the saturation channel
    hist_sat = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    
    hist_sat_smoothed = smooth_histogram_with_gaussian(hist_sat, sigma = 3)
    
    # Invert the histogram to find valleys as peaks
    relevant_histogram = hist_sat_smoothed[saturation_range[0]:saturation_range[1]+1]
    inverted_histogram = -relevant_histogram

    # Use find_peaks to find the indexes of these peaks (valleys in the original histogram)
    
    peaks, _ = find_peaks(inverted_histogram)

    # plot_histogram(inverted_histogram, "saturation")
    
    if peaks.size > 0:
        return peaks[0] + saturation_range[0] + sat_offset

    else:
        return 100
    