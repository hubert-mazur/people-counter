import argparse

import cv2
import imutils
import numpy as np
from imutils.object_detection import non_max_suppression

from progress import progress_bar

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--video", required=True, help="path to video")
filename = vars(ap.parse_args())['video']
video = cv2.VideoCapture(filename)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

max_width = 1000
max_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

ret, frame = video.read()
frame = imutils.resize(frame, width=min(max_width, frame.shape[1]))
height, width, channels = frame.shape
writer = cv2.VideoWriter('output/output.mov', cv2.VideoWriter_fourcc(*'XVID'), 16, (width, height))

i = 0
print('Processing video, this may take a while...')
progress_bar(i, max_frame, prefix='Processing video:', suffix='Complete')
while ret:
    frame = imutils.resize(frame, width=min(max_width, frame.shape[1]))
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    cv2.putText(frame, f'Total Persons : {len(rects)}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    writer.write(frame)

    i += 1
    progress_bar(i, max_frame, prefix='Processing video:', suffix='Complete')
    ret, frame = video.read()

print('Video has been processed correctly. Result file is located in output directory.')
video.release()
writer.release()
