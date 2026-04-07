import cv2
import numpy as np

cap = cv2.VideoCapture(0)
key = ''
while key != 'q':
    ret, src = cap.read()
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    # create mask
    mask1 = cv2.inRange(hsv, (36,50,70), (89,255,255))

    # apply mask
    cv2.bitwise_and(src, src, mask=mask1)

    cv2.imshow('output', mask1)
    key = cv2.waitKey(1)
cv2.destroyAllWindows()