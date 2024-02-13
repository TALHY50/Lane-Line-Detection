import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def calculate_fit(x1, x2, y1, y2):
    return np.polyfit((x1, x2), (y1, y2), 1)

def canny_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if len(gray.shape) != 2:
        raise ValueError("Image needs to be grayscale for Canny edge detection.")
    if gray.dtype != np.uint8:
        gray = np.uint8(gray / np.max(gray) * 255)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    binary_img = cv2.adaptiveThreshold(blur_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(binary_img, low_threshold, high_threshold)
    edges[edges > 0] = 255
    return edges

def adjust_night_vision(image, low_light_threshold=50):
    yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
    output_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)
    thresholds = (output_image[:,:,0] < low_light_threshold) | \
                 (output_image[:,:,1] < low_light_threshold) | \
                 (output_image[:,:,2] < low_light_threshold)
    output_image[thresholds] = [0, 0, 0]
    return output_image

def determine_day_or_night(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    average_brightness = np.mean(gray)
    brightness_threshold = 100
    return average_brightness > brightness_threshold

def adjust_roi_for_day(image):
    ysize = image.shape[0]
    xsize = image.shape[1]
    left_bottom = [1, ysize]
    right_bottom = [xsize, ysize]
    left_top = [xsize * 0.45, ysize * 0.6]
    right_top = [xsize * 0.55, ysize * 0.6]
    return left_bottom, right_bottom, left_top, right_top

def adjust_roi_for_night(image):
    ysize = image.shape[0]
    xsize = image.shape[1]
    left_bottom = [1, ysize]
    right_bottom = [xsize, ysize]
    left_top = [xsize * 0.40, ysize * 0.65]
    right_top = [xsize * 0.60, ysize * 0.65]
    return left_bottom, right_bottom, left_top, right_top

def process_image(image_path):
    image = mpimg.imread(image_path)
    original_image = np.copy(image)
    is_day = determine_day_or_night(image)
    if is_day:
        left_bottom, right_bottom, left_top, right_top = adjust_roi_for_day(image)
    else:
        image = adjust_night_vision(image)
        left_bottom, right_bottom, left_top, right_top = adjust_roi_for_night(image)
    ysize = image.shape[0]
    xsize = image.shape[1]
    color_select = np.copy(image)
    line_image = np.copy(image)
    red_threshold = 200
    green_threshold = 200
    blue_threshold = 200
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]
    thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                 (image[:,:,1] < rgb_threshold[1]) | \
                 (image[:,:,2] < rgb_threshold[2])
    color_select[thresholds] = [0, 0, 0]
    plt.imshow(color_select)
    plt.title(f"Color Selection for {image_path}")
    plt.show()
    edges = canny_edge_detection(image)
    plt.imshow(edges, cmap='Greys_r')
    plt.title(f"Canny Edge Detection for {image_path}")
    plt.show()
    # Create the mask for the region of interest
    region_mask = np.zeros_like(image)
    cv2.fillPoly(region_mask, np.int32([[
        left_bottom,
        left_top,
        right_top,
        right_bottom
    ]]), (255, 255, 255))
    masked_image = cv2.bitwise_and(original_image, region_mask)
    plt.imshow(masked_image)
    plt.title("Region of Interest Masked Image")
    plt.show()
    # Shifting the top vertices closer to the center to reduce the diameter
    left_bottom = [0, ysize]
    right_bottom = [xsize, ysize]
    left_top = [xsize*0.45, ysize*0.6]  # Adjusted from 0.4 to 0.45
    right_top = [xsize*0.55, ysize*0.6]  # Adjusted from 0.6 to 0.55

      # Four sides of trapezoid using optimized calculate_fit function
    fit_left = calculate_fit(left_bottom[0], left_top[0], left_bottom[1], left_top[1])
    fit_right = calculate_fit(right_bottom[0], right_top[0], right_bottom[1], right_top[1])
    fit_bottom = calculate_fit(left_bottom[0], right_bottom[0], left_bottom[1], right_bottom[1])
    fit_top = calculate_fit(left_top[0], right_top[0], left_top[1], right_top[1])

    # Create meshgrid
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    
    # Mask pixels for the green continuous lane lines (as before)
    region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                        (YY > (XX*fit_right[0] + fit_right[1])) & \
                        (YY < (XX*fit_bottom[0] + fit_bottom[1])) & \
                        (YY > (XX*fit_top[0] + fit_top[1]))

    line_image[~thresholds & region_thresholds] = [0, 255, 0]

    # Mask the region of interest on the image
    region_select = np.copy(line_image)
    region_select[~region_thresholds] = [0, 0, 0]
 # Plot the ROI with adjusted vertices
    plt.imshow(image)
    x = [left_bottom[0], right_bottom[0], right_top[0], left_top[0], left_bottom[0]]
    y = [left_bottom[1], right_bottom[1], right_top[1], left_top[1], left_bottom[1]]
    plt.plot(x, y, 'r--', lw=4)  
    plt.title("Adjusted Region Of Interest")
    plt.show()    

    plt.imshow(line_image)
    plt.title(f"Region Masked Image [Lane Lines in Green] for {image_path}")
    plt.show()
    # Convert only the green lanes to grayscale
    green_lanes = np.copy(line_image)
    green_lanes[(green_lanes[:, :, 0] != 0) | (green_lanes[:, :, 1] != 255) | (green_lanes[:, :, 2] != 0)] = [0, 0, 0]
    gray = cv2.cvtColor(green_lanes, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
   # Adjusted Canny edge detection
    edges = cv2.Canny(blurred_gray, 50, 150)
# HoughLinesP parameters
    rho = 1
    theta = np.pi / 180
    threshold = 30  # Adjusted as needed
    min_line_length = 10
    max_line_gap = 10

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    # Create an empty mask to draw lines
    line_mask = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_mask, (x1, y1), (x2, y2), (0, 255, 0), 15)  # Red color, increased thickness
    else:
        print(f"No lines were detected in {image_path}")

    # Dilate the mask to fill gaps
    kernel = np.ones((20, 20), np.uint8)  # Increased kernel size
    dilated_mask = cv2.dilate(line_mask, kernel, iterations=1)

    # Adjust the blending weights to make the lines more prominent
    result_image = cv2.addWeighted(image, 1, dilated_mask, 1, 0)

    plt.imshow(result_image)
    plt.title(f"Hough Transform for {image_path}")
    plt.show()
# List of image paths. Replace these with your paths
image_paths = ['/content/IMG_4089.jpg']

# Process each image
for img_path in image_paths:
    process_image(img_path)