import cv2
import numpy as np
import matplotlib.pyplot as plt

# Kalman Filter Setup
kalman = cv2.KalmanFilter(4, 2)  # 4 states (x, y, dx, dy), 2 measurements (x, y)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0]], dtype=np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                     [0, 1, 0, 1],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]], dtype=np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

# Load the video
video_path = './one_cycle.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video frame dimensions and FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize VideoWriter to save the detected video
output_video_path = './tracked_particle.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Parameters
distance_threshold = 50  # Maximum allowable distance between consecutive points
coordinates = []
prev_detected_point = None  # Store the last valid detected point

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV and threshold for orange color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([10, 100, 100])  # Adjust these values
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 10:  # Ignore small contours
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cy_bottom_left = frame_height - cy  # Adjust for bottom-left origin

                # Update Kalman Filter
                measurement = np.array([[np.float32(cx)], [np.float32(cy_bottom_left)]])
                kalman.correct(measurement)

                # Predict the next position
                prediction = kalman.predict()
                px, py = int(prediction[0]), int(prediction[1])

                # Compute distance between previous and current detected points
                if prev_detected_point:
                    dist = np.sqrt((cx - prev_detected_point[0]) ** 2 + (cy_bottom_left - prev_detected_point[1]) ** 2)
                else:
                    dist = 0

                # Accept the point only if it is within the threshold
                if dist <= distance_threshold or prev_detected_point is None:
                    coordinates.append((cx, cy_bottom_left))
                    prev_detected_point = (cx, cy_bottom_left)

                    # Draw the detected and predicted points
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # Detected point (green)
                    cv2.circle(frame, (px, frame_height - py), 5, (255, 0, 0), -1)  # Predicted point (blue)

    # Slow down visualization (optional)
    cv2.waitKey(10)  # Pause to slow down video playback

    # Write the frame to the output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Extract x and y coordinates
coordinates = np.array(coordinates)
x_data, y_data = coordinates[:, 0], coordinates[:, 1]

# Plot the trajectory
plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data, color="orange", label="Detected Trajectory")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate (Bottom-Left Origin)")
plt.title("Particle Trajectory with Distance Filtering")
plt.legend()
plt.grid()

# Save the plot and data
plt.savefig('filtered_trajectory.png')
np.savetxt("filtered_coordinates.csv", coordinates, delimiter=",", header="X,Y", comments="")
print(f"Coordinates saved to 'filtered_coordinates.csv'")
print(f"Tracked video saved to '{output_video_path}'")
