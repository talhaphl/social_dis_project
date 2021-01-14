import numpy as np
import argparse
import cv2
import imutils
import math

def rectangle_center(x,w,y,h):
    center_x =x +(w / 2)
    center_y =y+(h / 2)
    return(center_x,center_y)

def detect(frame):
    bounding_box_cordinates, weights = HOGCV.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)

    person = 1
    average_w = 0
    total_w = 0
    centers = []
    social_distancing_warning = []

    for x, y, w, h in bounding_box_cordinates:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        centers.append(rectangle_center(x,w,y,h))
        total_w = total_w + w
        cv2.putText(frame, f'person {person}', (x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        person += 1

    average_w = total_w / person

    for i, M in zip(range(len(centers)), centers):
        for j, N in zip(range(len(centers)), centers[i + 1:]):
            if math.sqrt(math.pow((M[0] - N[0]),2) + (math.pow((M[1] - N[1]),2))) < average_w:
                social_distancing_warning.append((i,j + i + 1))

    for (i,j) in social_distancing_warning:
        (x, y, w, h) = bounding_box_cordinates[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        (x, y, w, h) = bounding_box_cordinates[j]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, '!! DANGEROUS DISTANCE !!', (x, y - 8), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255), 1)

    cv2.putText(frame, 'Status : Detecting ', (30, 420), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f'Total Persons : {person - 1}', (30, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 1)
    cv2.imshow('output', frame)

    return frame

def detectByPathVideo(path):

    video = cv2.VideoCapture(path)
    check, frame = video.read()

    if check == False:
        print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
        return
    print('Detecting people...')

    while video.isOpened():
        check, frame = video.read()

        if check:
            frame = imutils.resize(frame, width=min(800, frame.shape[1]))
            frame = detect(frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()


path = "test.mp4"
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

detectByPathVideo(path)

