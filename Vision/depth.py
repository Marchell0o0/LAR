import cv2
import numpy as np
import math

def generate_pc_image(pc_image):
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
    
    return

def generate_deth_image(depth_image):
    # Scale the depth image to the full range of 8-bit color scale
    max_val = np.max(depth_image)
    min_val = np.min(depth_image)
    scaled_depth = ((depth_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    # Mask out the depths that are outside the specified range
    scaled_depth[depth_image == 0] = 0  # Assuming 0 is the value for invalid depth information

    # Apply a colormap to the scaled depth image
    colored_depth = cv2.applyColorMap(scaled_depth, cv2.COLORMAP_JET)
    
    return

def find_point_depth(depth_image, x, y):
    depth_value = depth_image[y, x] 
    
    return depth_value
    
def find_point_pc_coords(pc_image,
                         x,
                         y,
                         y_offset=-0.04):
    array_shape = pc_image.shape
    height = array_shape[0]
    width = array_shape[1]
    
    if y < 0 or y >= height or x < 0 or x >= width:
        return None, None, None
    
    y_value = pc_image[y, x, 0] + y_offset
    x_value = pc_image[y, x, 1] 
    z_value = pc_image[y, x, 2]

    return x_value, y_value, z_value