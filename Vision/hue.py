import cv2
import numpy as np

from Vision.utils import smooth_histogram_with_gaussian

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
    
    return int(average_hue)

def calculate_hue_peak_average(image, color_range, saturation_threshold, min_peak=0.02):
    """Choose the weighted value for hue based on peaks that correspond to the saturation value"""
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_hue = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()

    hist_hue_smoothed = smooth_histogram_with_gaussian(hist_hue, sigma = 3)
    
    # Normalize histogram
    hist_norm = hist_hue_smoothed / hist_hue_smoothed.max()

    # utils.plot_histogram(hist_norm, "Normalized hue histogram")
    
    # Find potential peaks
    peaks = [i for i in range(color_range[0], color_range[1]+1) if hist_norm[i] > min_peak]

    # Validate peaks based on saturation value
    valid_peaks = []
    for peak in peaks:
        sat_values = hsv[:,:,1][hsv[:,:,0] == peak]
        if sat_values.size > 0 and np.mean(sat_values) > saturation_threshold:
            valid_peaks.append(peak)

    # Calculate weighted average if multiple valid peaks
    if valid_peaks:
        weighted_avg = np.average(valid_peaks, weights=[hist_norm[peak] for peak in valid_peaks])
        return int(weighted_avg)
    else:
        return calculate_hue_average(hist_hue_smoothed, color_range)


def calculate_hue_params(image, color_range, deviation_range, saturation_threshold, color_name):
    
    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate the histogram for the hue channel
    hist_hue = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    
    hist_hue_smoothed = smooth_histogram_with_gaussian(hist_hue, sigma = 3)
    
    
    average_hue = calculate_hue_peak_average(image, color_range, saturation_threshold)
    
    # Calculate standard deviation around the peak
    color_deviation = np.std(hsv[:,:,0][(hsv[:,:,0] > average_hue - deviation_range) & (hsv[:,:,0] < average_hue + deviation_range)])

    return average_hue, color_deviation, hist_hue_smoothed