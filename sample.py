import cv2 as cv
img = cv.imread("./random_image.png")

cv.imshow("Display window", img)
k = cv.waitKey(0)