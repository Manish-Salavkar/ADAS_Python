import cv2
import numpy as np

def detect_lanes(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Define a region of interest (ROI)
    height, width = edges.shape[:2]
    roi_vertices = np.array([[(0, height), (width // 2, height // 2), (width, height)]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Perform Hough line detection
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=50)
    
    # Draw the detected lane lines on the image
    line_image = np.zeros_like(image)
    draw_lines(line_image, lines)
    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    
    return result

def draw_lines(image, lines, color=(0, 0, 255), thickness=3):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)

# Load an image
image_path = "image.jpg"
image = cv2.imread(image_path)

# Detect lanes
result_image = detect_lanes(image)

# Display the original and result images
# cv2.imshow("Original Image", image)
cv2.imshow("Lane Detection Result", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
