# import the necessay packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# parse the arguements
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper bounds
lower = (29, 86, 6)
upper = (64, 255, 255)
pts = deque(maxlen=args["buffer"])

# grab the stream
if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])

# allow the video/camera to warm up
time.sleep(2.0)

# keep looping
while True:
    # grab the frame
    frame = vs.read()

    # handle the frame
    frame = frame[1] if args.get("video", False) else frame

    # if we did not grab a frame the video is done
    if frame is None:
        break

    # resize, blur, and convert the frame
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # create a mask, dilate and erode it
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours and center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only process found contours
    if len(cnts) > 0:
        # find largest contour and compute centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # proceed if radius meets minimum size
        if radius > 10:
           # draw circle and centroid on the frame
           cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
           cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # update the point queue
    pts.appendleft(center)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # ignore non points
        if pts[i - 1] is None or pts[i] is None:
            continue

        # draw connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # show the frame
    cv2.imshow("Frame", frame)

    # stop on q press
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break 

# stop the stream
if not args.get("video", False):
    vs.stop()
else:
    vs.release()

# destroy all windows
cv2.destroyAllWindows()
