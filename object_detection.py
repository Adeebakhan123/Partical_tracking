import cv2
import numpy as np

# Load the video or start webcam (if you want to use real-time camera feed)
cap = cv2.VideoCapture('./jupiter_timelapse.mp4')

# Define the range for nearly white color in HSV space
lower_white = np.array([0, 0, 200])  # Lower bound for white (low saturation and high value)
upper_white = np.array([180, 30, 255])  # Upper bound for white (low saturation and high value)

delay = 100

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("End of video or failed to load.")
        break
    
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a binary mask where the white color falls within the range
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Perform some morphological operations to remove noise and smooth the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # Find contours (boundaries of the white object in the mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If any contours are found
    if contours:
        # Get the largest contour, assuming it's Jupiter
        c = max(contours, key=cv2.contourArea)
        
        # Get the center of the contour (centroid)
        M = cv2.moments(c)
        if M["m00"] > 0:  # Avoid division by zero
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Draw the center on the frame
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            
            # Print the coordinates of Jupiter
            print(f"Jupiter coordinates: ({cx}, {cy})")
    
    # Display the frame with Jupiter marked
    cv2.imshow('Jupiter Tracking', frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
