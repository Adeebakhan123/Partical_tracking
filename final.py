import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the video
video_path = './one_cycle.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Coordinates storage
coordinates = []

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for orange color (Adjust based on the video)
    lower_orange = np.array([10, 100, 100])  # Example lower HSV range for orange
    upper_orange = np.array([25, 255, 255]) # Example upper HSV range for orange

    # Threshold the HSV image to get only orange colors
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Assume the largest contour is the particle
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 10:  # Ignore small areas as noise
            # Get the center of the contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Shift origin to bottom-left
                cy_bottom_left = frame_height - cy
                coordinates.append((cx, cy_bottom_left))

                # Draw the detected point on the frame
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    # Display the frame (optional for visualization)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Extract x and y coordinates
coordinates = np.array(coordinates)
x_data, y_data = coordinates[:, 0], coordinates[:, 1]

# Plot only the points
plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data, color="orange", label="Particle Positions")

plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate (Bottom-Left Origin)")
plt.title("Orange Particle Trajectory")
plt.legend()
plt.grid()

# Save plot to file
plt.savefig('particle_positions.png')

# Save coordinates to a file
output_path = "./coordinates_only_points.csv"
np.savetxt(output_path, coordinates, delimiter=",", header="X,Y", comments="")
print(f"Coordinates saved to {output_path}")
