import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# Load the video
cap = cv2.VideoCapture('WhatsApp Video 2024-11-10 at 6.16.12 PM.mp4')

# Define the HSV range for the orange particle
lower_orange = np.array([10, 100, 100])  
upper_orange = np.array([25, 255, 255])

coordinates = []

# Open a CSV file to save the coordinates
with open('jupiter_coordinates.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame', 'X', 'Y'])  
    
    frame_count = 0
    delay = 1
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("End of video or failed to load.")
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Apply morphological operations
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            
            M = cv2.moments(c)
            if M["m00"] > 0:  
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Append the coordinates
                coordinates.append((frame_count, cx, cy))
                
                # Write coordinates to CSV
                writer.writerow([frame_count, cx, cy])
                
                # Draw the detected particle center
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        
        # Display the tracking frame
        cv2.imshow('Jupiter Tracking', frame)
        
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        
        frame_count += 1

# Release resources
cap.release()
cv2.destroyAllWindows()

# Process coordinates for best-fit trajectory
if coordinates:
    # Extract frame (time), X, and Y values
    frames, x_coords, y_coords = zip(*coordinates)
    
    # Fit a polynomial (e.g., degree 4 for complex paths) to the trajectory
    poly_x = Polynomial.fit(frames, x_coords, deg=5)
    poly_y = Polynomial.fit(frames, y_coords, deg=5)
    
    # Get the polynomial coefficients for X and Y
    x_coef = poly_x.convert().coef
    y_coef = poly_y.convert().coef
    print("Best-fit polynomial coefficients for X:", x_coef)
    print("Best-fit polynomial coefficients for Y:", y_coef)
    
    # Generate fitted trajectory values based on the polynomial equation
    fitted_frames = np.linspace(frames[0], frames[-1], 100)  # Interpolated frames for smooth curve
    fitted_x = sum(c * fitted_frames**i for i, c in enumerate(x_coef))  # Apply X polynomial
    fitted_y = sum(c * fitted_frames**i for i, c in enumerate(y_coef))  # Apply Y polynomial
    
    # Plotting the original coordinates and the fitted trajectory
    plt.figure()
    plt.scatter(x_coords, y_coords, color='blue', label='Detected Particle Position')
    plt.plot(fitted_x, fitted_y, color='red', label='Best-Fit Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Particle Trajectory with Best-Fit Curve')
    plt.legend()
    plt.savefig('predicted_trajectory.png')  # Save the plot as an image file

    # plt.show()

    # Store the predicted trajectory in a CSV file
    with open('predicted_trajectory.csv', mode='w', newline='') as traj_file:
        traj_writer = csv.writer(traj_file)
        traj_writer.writerow(['Frame', 'Predicted X', 'Predicted Y'])
        for f, x, y in zip(fitted_frames, fitted_x, fitted_y):
            traj_writer.writerow([int(f), int(x), int(y)])

else:
    print("No coordinates found to fit trajectory.")
