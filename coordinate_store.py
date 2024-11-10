import cv2
import numpy as np
import csv

cap = cv2.VideoCapture('./jupiter_timelapse.mp4')

lower_white = np.array([0, 0, 200]) 
upper_white = np.array([180, 30, 255])  
coordinates = []

with open('jupiter_coordinates.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame', 'X', 'Y'])  
    
    frame_count = 0
    delay = 100  
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("End of video or failed to load.")
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        cv2.imshow('Mask', mask)
        
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"Frame {frame_count}: {len(contours)} contours detected")
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            
            M = cv2.moments(c)
            if M["m00"] > 0:  
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                coordinates.append((frame_count, cx, cy))
                
                writer.writerow([frame_count, cx, cy])
                
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                
                print(f"Frame {frame_count}: Jupiter coordinates: ({cx}, {cy})")
        
        cv2.imshow('Jupiter Tracking', frame)
        
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        
        frame_count += 1

cap.release()
cv2.destroyAllWindows()

with open('jupiter_coordinates_list.txt', mode='w') as file:
    for coord in coordinates:
        file.write(f"Frame {coord[0]}: Jupiter coordinates: ({cx}, {cy})\n")

print("Tracked coordinates of Jupiter:", coordinates)


# import cv2
# import numpy as np
# import csv

# # Load the video
# cap = cv2.VideoCapture('path_to_video_of_jupiter.mp4')

# # Define the range for nearly white color in HSV space (adjust as needed)
# lower_white = np.array([0, 0, 200])  # Lower bound for white
# upper_white = np.array([180, 30, 255])  # Upper bound for white

# # List to store the coordinates of Jupiter
# coordinates = []

# # Open a CSV file to save the coordinates
# with open('jupiter_coordinates.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Frame', 'X', 'Y'])  # CSV header
    
#     frame_count = 0
#     delay = 100  # Delay between frames (100 ms, adjust as needed)
    
#     while cap.isOpened():
#         ret, frame = cap.read()
        
#         if not ret:
#             print("End of video or failed to load.")
#             break
        
#         # Convert the frame to HSV color space
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
#         # Create a binary mask where the white color falls within the range
#         mask = cv2.inRange(hsv, lower_white, upper_white)
        
#         # Display the mask to check if the white object is being detected
#         cv2.imshow('Mask', mask)
        
#         # Perform some morphological operations to remove noise and smooth the mask
#         mask = cv2.erode(mask, None, iterations=2)
#         mask = cv2.dilate(mask, None, iterations=2)
        
#         # Find contours (boundaries of the white object in the mask)
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         # Print the number of contours detected for debugging
#         print(f"Frame {frame_count}: {len(contours)} contours detected")
        
#         # If any contours are found
#         if contours:
#             # Get the largest contour, assuming it's Jupiter
#             c = max(contours, key=cv2.contourArea)
            
#             # Get the center of the contour (centroid)
#             M = cv2.moments(c)
#             if M["m00"] > 0:  # Avoid division by zero
#                 cx = int(M["m10"] / M["m00"])
#                 cy = int(M["m01"] / M["m00"])
                
#                 # Store the coordinates in the list
#                 coordinates.append((frame_count, cx, cy))
                
#                 # Write the coordinates to the CSV file
#                 writer.writerow([frame_count, cx, cy])
                
#                 # Draw the center on the frame
#                 cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                
#                 # Print the coordinates of Jupiter
#                 print(f"Frame {frame_count}: Jupiter coordinates: ({cx}, {cy})")
        
#         # Display the frame with Jupiter marked
#         cv2.imshow('Jupiter Tracking', frame)
        
#         # Slow down the video by increasing the delay
#         if cv2.waitKey(delay) & 0xFF == ord('q'):
#             break
        
#         # Increment frame count
#         frame_count += 1

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# # Optionally, you can print or analyze the stored coordinates list
# print("Tracked coordinates of Jupiter:", coordinates)
