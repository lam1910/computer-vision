import imutils
import cv2
import numpy as np


video_path = './data/cam125.avi'

# Initialize video reader object
vs = cv2.VideoCapture(video_path)

# Initialize GMM object
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=True)

# List detection by background subtraction
list_detection = []

# Process
while True:
    # Read video
    _, frame = vs.read()

    # If end of video, break
    if frame is None:
        break

    # resize the frame, easy to display
    frame = imutils.resize(frame, width=500)

    # Apply GMM model to frame
    thresh = fgbg.apply(frame)

    # Remove noise
    thresh = cv2.medianBlur(thresh, 5)

    # Dilate
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Threshold image
    _, thresh = cv2.threshold(thresh, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    _, cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find contours with maximum area
    maxArea = 0
    maxContour = None
    for c in cnts:
        if (cv2.contourArea(c) > maxArea) and (cv2.contourArea(c) >= 500):
            maxArea = cv2.contourArea(c)
            maxContour = c

    if maxContour is None:
        list_detection = []

    else:
        # Draw result
        (x, y, w, h) = cv2.boundingRect(maxContour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        dx = int(x + w / 2)
        dy = int(y + h / 2)
        list_detection.append((dx, dy))

        if len(list_detection) > 1:
            for i in range(len(list_detection) - 1):
                x, y = list_detection[i]
                u, v = list_detection[i + 1]
                cv2.line(frame, (x, y), (u, v), (0, 0, 255))

    # Display result
    cv2.imshow("Original", frame)
    cv2.imshow("GMM", thresh)

    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
