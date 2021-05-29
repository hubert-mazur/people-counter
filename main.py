import numpy as np
from imutils.object_detection import non_max_suppression
import imutils
import argparse
import cv2

WIDTH = 1000

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--video", required=True, help="path to video")
filename = vars(ap.parse_args())['video']

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

video = cv2.VideoCapture(filename)
ret, frame = video.read()

frame = imutils.resize(frame, width=min(WIDTH, frame.shape[1]))
height, width, channels = frame.shape
writer = cv2.VideoWriter('output/output.mov', cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))


i = 1
while ret:
    print(f'processing {i} frame...')
    i += 1

    frame = imutils.resize(frame, width=min(WIDTH, frame.shape[1]))
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    writer.write(frame)
    # cv2.imwrite(f'test/{i}.png', frame)
    ret, frame = video.read()

video.release()
writer.release()
