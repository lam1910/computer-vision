import imutils
import cv2
import numpy as np


# Create kalman filter object
def create_kalman():
    dt = 0.2
    kalman = cv2.KalmanFilter(4, 2, 0)
    kalman.transitionMatrix = np.array(
        [[1., 0., dt, 0.], [0., 1., 0., dt], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype=np.float32)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kalman.processNoiseCov = 0.5 * np.array([[dt ** 4.0 / 4.0, 0., dt ** 3.0 / 2.0, 0.],
                                             [0., dt ** 4.0 / 4.0, 0., dt ** 3.0 / 2.0],
                                             [dt ** 3.0 / 2.0, 0., dt ** 2.0, 0.],
                                             [0., dt ** 3.0 / 2.0, 0., dt ** 2.0]], dtype=np.float32)
    kalman.measurementNoiseCov = 1e-1 * np.eye(2, dtype=np.float32)
    kalman.errorCovPost = 1e-1 * np.eye(4, dtype=np.float32)

    return kalman


video_path = './data/cam125.avi'

# Initialize video reader object
vs = cv2.VideoCapture(video_path)

# Initialize GMM object
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=True)

# Kalman Filter
kalman = create_kalman()

# List detection by background subtraction
list_detection = []

# List estimation by kalman filter
list_predict = []

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

        # New
        list_predict = []

    else:
        # Draw result
        (x, y, w, h) = cv2.boundingRect(maxContour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        dx = int(x + w / 2)
        dy = int(y + h / 2)
        list_detection.append((dx, dy))

        # Init statePre
        if len(list_detection) == 1:
            kalman = create_kalman()
            kalman.statePre = np.array([[dx], [dy], [0.], [0.]], dtype=np.float32)

        # Kalman correct
        kalman.correct(np.array([[dx], [dy]], dtype=np.float32))

        # Kalman predict
        estimate = kalman.predict()
        list_predict.append((estimate[0, 0], estimate[1, 0]))

        if len(list_detection) > 1:
            for i in range(len(list_detection) - 1):
                x, y = list_detection[i]
                u, v = list_detection[i + 1]
                cv2.line(frame, (x, y), (u, v), (0, 0, 255))

            # Draw tracker
            for i in range(len(list_detection) - 1):
                x, y = list_predict[i]
                u, v = list_predict[i + 1]
                cv2.line(frame, (x, y), (u, v), (255, 0, 0))

    # Display result
    cv2.imshow("Original", frame)
    cv2.imshow("GMM", thresh)

    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
