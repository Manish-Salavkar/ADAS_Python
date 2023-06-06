import cv2
import numpy as np

def detect_lanes(image):
    # Same lane detection code as before...
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

# Open the video file
video_path = "lane_vid.mp4"
cap = cv2.VideoCapture(video_path)

# Get the video's frames per second (fps) and frame size
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object to save the processed video
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Detect lanes in the current frame
    result_frame = detect_lanes(frame)
    
    # Write the processed frame to the output video
    output.write(result_frame)
    
    # Display the resulting frame (optional)
    cv2.imshow("Lane Detection", result_frame)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and VideoWriter objects
cap.release()
output.release()

# Close all windows
cv2.destroyAllWindows()
