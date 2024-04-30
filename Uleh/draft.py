# test functions -----------------------------------------

# def calculate_sat_average(hist_sat, saturation_range):
#     # Find the peak hue value within the range
#     argmax_saturation = np.argmax(hist_sat[saturation_range[0]:saturation_range[1]]) + saturation_range[0]
    
#     # Find the minimum hue value (valley) within the same range
#     argmin_saturation = np.argmin(hist_sat[saturation_range[0]:saturation_range[1]]) + saturation_range[0]
#     # Calculate the average of the peak and valley
#     sat_threshold = (argmax_saturation + argmin_saturation) // 2
#     return sat_threshold

# def find_peak_ranges(hist, threshold=0.1, min_range=10):
#     """Find peaks in histogram and their corresponding ranges."""
#     peaks = []
#     start = None

#     for i in range(1, len(hist) - 1):
#         # If the histogram value is above the threshold and rising
#         if hist[i] > threshold * np.max(hist) and hist[i] > hist[i - 1]:
#             if start is None:
#                 start = i
#         # If the histogram value is above the threshold and falling
#         elif start is not None and hist[i] > threshold * np.max(hist) and hist[i] < hist[i + 1]:
#             # If the range is wide enough, consider it a peak
#             if i - start > min_range:
#                 peak = (start, i)
#                 peaks.append(peak)
#             start = None

#     return peaks

# def adaptive_threshold_hsv(image, color='green'):
#     """TEST inRange: Adaptively create thresholds for the specified color in an HSV image."""
#     # Convert to HSV
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
#     # Define color ranges for green
#     if color == 'green':
#         hue_range = (40, 80)  # Typical green hue range
#     else:
#         raise ValueError("Color not supported.")

#     # Calculate histograms
#     hist_hue = cv2.calcHist([hsv], [0], None, [180], hue_range)
#     hist_hue = cv2.normalize(hist_hue, hist_hue).flatten()
#     hist_sat = cv2.calcHist([hsv], [1], None, [256], [0, 256])
#     hist_sat = cv2.normalize(hist_sat, hist_sat).flatten()

#     # Find the peak ranges for hue and saturation
#     hue_peaks = find_peak_ranges(hist_hue)
#     sat_peaks = find_peak_ranges(hist_sat)

#     # Check if peaks were found; if not, default to some value or raise an error
#     if not hue_peaks:
#         raise ValueError("No hue peaks found.")
#     if not sat_peaks:
#         raise ValueError("No saturation peaks found.")

#     # Use the first peak for simplicity
#     hue_lower = hue_peaks[0][0]
#     hue_upper = hue_peaks[0][1]
#     sat_lower = sat_peaks[0][0]
#     sat_upper = 255  # Usually, high saturation values represent the color more accurately

#     lower_bound = np.array([hue_lower, sat_lower, 50])  # 50 is an arbitrary value for V
#     upper_bound = np.array([hue_upper, sat_upper, 255])  # 255 for max V value

#     return lower_bound, upper_bound

# use this function for a green obstacle
# def calculate_hue_params_green(image, color_range):
#     # Convert image to HSV
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
#     # Calculate the histogram for the hue channel
#     hist_hue = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    
#     # hist_hue_smoothed = smooth_histogram(hist_hue, window_size=10)
#     hist_hue_smoothed = utils.smooth_histogram_with_gaussian(hist_hue, sigma = 3)
    
#     peak_green_hue = np.argmax(hist_hue_smoothed[color_range[0]:color_range[1]]) + color_range[0]

#     # Threshold the image to get a mask for green areas
#     lower_green = np.array([peak_green_hue - 20, 40, 10])
#     upper_green = np.array([peak_green_hue + 20, 255, 255])
    
#     # WARNING: does not work
#     # lower_green, upper_green = adaptive_threshold_hsv(image)
    
#     mask_green = cv2.inRange(image, lower_green, upper_green)
    
#     # Debugging
#     # print("Peak green value: ", peak_green_hue)
    
#     return mask_green

# test functions -----------------------------------------