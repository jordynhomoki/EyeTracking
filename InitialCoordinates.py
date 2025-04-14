'''
Using OpenCV to attain the initial coordinates of an object to then use as inputs for VWM_EyeTracking.py,
HND_EyeTracking.py.
'''

import cv2
import sys
from AudioDetection import blockStart


def click_event(event, x, y, flags, params):
    # function used to click on the first frame of a video in order to return the corresponding coordinates

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates on the Shell
        coordinates.append(x)
        coordinates.append(y)

        # displaying the coordinates on the image window
        cv2.circle(frame, (x, y), 1, (255, 0, 0))
        cv2.imshow('Frame', frame)


# open the video file; modify run configuration to input the path to any video file
cap = cv2.VideoCapture(sys.argv[1])

while True:
    # read only the first frame of the inputted video, reconstruct frame parameters
    ret, frame = cap.read()

    # create list to store coordinates
    coordinates = []

    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    inBlock = False
    if abs(t - blockStart) <= 0.005:
        inBlock = True
    if not inBlock:
        continue

    cv2.imshow('Frame', frame)
    print("\nPlease left-click at the approximate center of the face and press Enter.")
    cv2.setMouseCallback('Frame', click_event)

    # press any key to escape
    key = cv2.waitKey(0)
    if key == ord('\r'):
        break

# releasing video capture
cap.release()
cv2.destroyAllWindows()
