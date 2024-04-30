import cv2
import numpy as np
import sys
import math



def is_circle(contour, circularity_threshold = 0.8):
    """
    Helper function for mark_polygons
    Don't want to mask out the circles so just skip the shapes that are too round

    Parameters:
    contour (array): The contour to evaluate.
    circularity_threshold (float): The threshold used to decide if the shape is a circle.

    Returns:
    bool: True if the shape is a circle, False otherwise.
    """
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)

    if perimeter == 0:
        return False
    
    circularity = 4 * math.pi * (area / (perimeter * perimeter))
    return circularity > circularity_threshold

def is_too_narrow(contour, aspect_ratio_threshold = 15):
    """
    Helper function for mark_polygons
    Wide lines were being classified as narrow polygons, so skip them with this function

    Parameters:
    contour (array): The contour to evaluate.
    aspect_ratio_threshold (float): The aspect ratio value above which the shape is considered too narrow.

    Returns:
    bool: True if the shape is too narrow, False otherwise.
    """
    # Calculate the rotated bounding rectangle of the contour
    rect = cv2.minAreaRect(contour)
    (width, height) = rect[1]

    width, height = max(width, height), min(width, height)

    if height == 0:
        return True
    
    aspect_ratio = width / height
    return aspect_ratio > aspect_ratio_threshold

def mark_circles(image, gray):
    """
    Detect and mark circles in an image.

    
    Parameters:
    image (array): The original image where circles will be marked.
    gray (array): Grayscale version of the image for processing.

    Returns:
    array: The image with detected circles marked.
    """

    """
    Possible changes 
    HOUGH_GRADIENT vs HOUGH_GRADIENT_ALT
    second one is said to have better accuracy

    HOUGH_GRADIENT params:
    ...

    HOUGH_GRADIENT_ALT params:
    minDist 50 ?
    param1 200 works well
    param2 0.9 works well
    minmaxRadius 0 ?
    """
    detected_circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 1, minDist=50,
                           param1=200, param2=0.9, minRadius=0, maxRadius=0)

    for circle in detected_circles[0, :]:  
        x, y, r = circle
        center = (int(x), int(y))
        radius = int(r)

        cv2.circle(image, center, radius, (0, 255, 0), 2)
        cv2.putText(image, 'Circle', (center[0] - 10, center[1]), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return image

def mark_polygons(image, contours, min_area = 500, epsilon = 0.02, max_vertices = 4):
    """
    Detect and label polygons in an image based on the number of vertices.
    Checks for: area, circularity, sides ratio for being too narrow, convexity, number of vertices

    Parameters:
    - image (array): The image on which to mark and label polygons.
    - contours (list of arrays): A list of contour points to evaluate.
    - min_area (float): The minimum area a contour must have to be considered a polygon.
    - epsilon (float): The approximation accuracy (as a proportion of the contour perimeter).
    - max_vertices (int): The maximum number of vertices for a contour to be labeled.

    Returns:
    array: The image with detected polygons marked.
    """
    for contour in contours:
        if cv2.contourArea(contour) < min_area or is_too_narrow(contour):
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon * peri, True)

        if not cv2.isContourConvex(approx) or len(approx) > max_vertices:
            continue

        M = cv2.moments(contour)

        # m00 will never be zero for polygons without self-intesections
        if M["m00"] == 0:
            continue

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        x,y,w,h = cv2.boundingRect(contour)
        
        num_vertices = len(approx)
        if num_vertices == 4:
            shape_name = "Rectangle"
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            cv2.putText(image, shape_name, (cX, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image


def detect_shapes(image_path):
    # Read an image and make a grayscale for circle detection
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply for noisy images, usually made by real cameras. For a vector image makes it worse
    # gray = cv2.GaussianBlur(gray, (9, 9), 0)

    # Find and display contours for polygon detection
    canny_output = cv2.Canny(gray, 20, 40)
    contours, _ = cv2.findContours(canny_output,
                                    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255, 0, 0), 2)
    
    image = mark_circles(image, gray)
    image = mark_polygons(image, contours)
                    
    cv2.imshow('Detected', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('detected_shapes.png', image)


if __name__ == "__main__":
    detect_shapes(sys.argv[1])




