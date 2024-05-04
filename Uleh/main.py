# import numpy as np # type: ignore
import cv2
import numpy as np

import utils
from robolab_turtlebot import Turtlebot, Rate, get_time     
  
from rectangle import RectangleProcessor
from color import ColorSettings

def main:
    color_settings = ColorSettings()
    color_settings.calibrate_color(turtle)
    
    # in while
    image = turtle.get_rgb_image()
    pc_image = turtle.get_point_cloud()
    
    rectg_processor = RectangleProcessor(image,
                                            pc_image,
                                            color_settings)
    detected_rectgs, masked_rectgs, image_rectg  = rectg_processor.process_image()

# def main():
#     turtle = Turtlebot(rgb = True, depth = True, pc = True)
    
#     color_settings = ColorSettings()
    
#     print('Waiting for rgb and point cloud ...')
#     turtle.wait_for_point_cloud()
#     turtle.wait_for_rgb_image()
#     print('First rgb and point cloud recieved ...')

#     # CALIBRATION
#     # -------------------------------------------
#     color_settings.calibrate_color(turtle)
#     # -------------------------------------------
    
#     # ADAPTATION
#     # -------------------------------------------
#     # color_settings.adapt_image_colors(turtle)
#     # -------------------------------------------
    
#     # OBSTACLE RECOGNITION
#     # -------------------------------------------
#     running = True
#     while running and not turtle.is_shutting_down():
        
#         image = turtle.get_rgb_image()
#         pc_image = turtle.get_point_cloud()
        
#         rectg_processor = RectangleProcessor(image,
#                                              pc_image,
#                                              color_settings)
#         detected_rectgs, masked_rectgs, image_rectg  = rectg_processor.process_image()
        
#         # Check image and pc_image resolution
#         # image_size = image.shape
#         # pc_image_size = pc_image.shape
        
#         # height_i = image_size[0]
#         # height_pc = pc_image_size[0]
#         # width_i = image_size[1]
#         # width_pc = pc_image_size[1]
    
#         # print(f"RGB image width is: {width_i}. height: {height_i}")
#         # print(f"RGB image width is: {width_pc}. height: {height_pc}")
        
#         cv2.imshow("RGB camera", image_rectg)
#         if (cv2.waitKey(1) & 0xFF == ord('q')) or turtle.is_shutting_down(): 
#             running = False
#     # -------------------------------------------    
        
# def main():
    
#     IMAGE_WIDTH = 1280
#     IMAGE_HEIGHT = 720
    
#     color_settings = ColorSettings()
    
#     # TEST FINDING DEPTH AND REAL COORDS
#     # -------------------------------------------
#     # Test Point Cloud:
#     # pc_image = np.load("test_images/spins/meas01/x01_1_pc.npy")
#     # depth.test_pc(pc_image)
    
#     # Test Depth image:
#     # depth_image = np.load("test_images/spins/meas02/x01_7_depth.npy")
#     # depth.generate_deth_image(depth_image)
    
#     # Test Point distance PC:
#     # pc_image = np.load("test_images/spins/meas01/x01_1_pc.npy")
#     # depth.find_point_pc_coords(pc_image, 635, 299)
    
#     # Test Point distance:
#     # depth_image = np.load("test_images/spins/meas02/x01_7_depth.npy")
#     # depth.find_point_depth(depth_image, 43, 446)
#     # -------------------------------------------
    
#     # CALIBRATE COLOR VALUES
#     # -------------------------------------------
#     image_names = []
    
#     # All 3 folders of photos
#     # for i in range(1, 4):
#     #     for j in range(1, 14):
#     #         image_names.append("test_images/spins/meas0{}/x01_{}_image.npy".format(i, j)) 
    
#     # Only 1 folder of photos
#     for j in range(1, 14):
#         image_names.append("../test_images/spins/meas01/x01_{}_image.npy".format(j))       
    
#     # Return dict with hue_avg, dev_avg and sat_avg for every color
#     color_settings.calibrate_color_debug(image_names)    
    
#     # -------------------------------------------
    
#     # TEST OBSTACLE RECOGNITION:
#     # -------------------------------------------
#     # One photo:
    
#     # image_name = "test_images/images/x01_image.npy"
#     # pc_image_name = "test_images/images/x01_pc.npy"
    
#     image_name = "../test_images/spins/meas01/x01_7_image.npy"
#     pc_image_name = "../test_images/spins/meas01/x01_7_pc.npy"
#     image = np.load(image_name)
#     pc_image = np.load(pc_image_name)
#     rectg_processor = RectangleProcessor(image,
#                                              pc_image,
#                                              color_settings)
#     detected_rectgs, masked_rectgs, image_rectg  = rectg_processor.process_image()
    
#     # Assign color value corresponding to the color of found obstacle
    
#     colors_reassigned_counter = 0
    
#     print("*****************************")
#     if colors_reassigned_counter < len(color_settings.colors):
#         for rectg in detected_rectgs:
#             color_name = utils.color_value_to_str(rectg.color)
#             if color_settings.calib_values[color_name]["color_reassigned"] == False:
#                 color_settings.assign_detected_color(rectg, image)
#                 colors_reassigned_counter += 1
#                 print(f"New average hue for {color_name}: {color_settings.calib_values[color_name]['hue_avg']}")
#                 print(f"New average sat for {color_name}: {color_settings.calib_values[color_name]['sat_avg']}")
#     print("*****************************")
    
#     color_queue = []
    
#     image_name = "../test_images/spins/meas01/x01_7_image.npy"
#     pc_image_name = "../test_images/spins/meas01/x01_7_pc.npy"
#     image = np.load(image_name)
#     pc_image = np.load(pc_image_name)
#     rectg_processor = RectangleProcessor(image,
#                                              pc_image,
#                                              color_settings)
#     detected_rectgs, masked_rectgs, image_rectg = rectg_processor.process_image()
    
#     # All photos:      
#     for i in range(1, 14):
#         image_name = "../test_images/spins/meas01/x01_{}_image.npy".format(i)
#         pc_image_name = "../test_images/spins/meas01/x01_{}_pc.npy".format(i)
#         image = np.load(image_name)
#         pc_image = np.load(pc_image_name)
#         rectg_processor = RectangleProcessor(image,
#                                              pc_image,
#                                              color_settings)
#         detected_rectgs, masked_rectgs, image_rectg = rectg_processor.process_image()
#         print(f"File no. {i}")
#     # -------------------------------------------

#     # GENERATE HSV DIAGRAMS
#     # -------------------------------------------
#     # utils.generate_HSV_image(image)
#     # utils.generate_general_HSV()
#     # -------------------------------------------

if __name__ == "__main__":
    main()