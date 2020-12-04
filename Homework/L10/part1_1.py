import imutils
import cv2
import numpy as np


video_path = './data/cam123.avi'

# Initialize video reader object
vs = cv2.VideoCapture(video_path)

# Initialize GMM object
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=True)

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

    # Display result
    cv2.imshow("Original", frame)
    cv2.imshow("GMM", thresh)

    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
