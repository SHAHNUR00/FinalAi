import cvzone
import cv2
import mediapipe as mp
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np
import streamlit as st

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=3)

textList = ["Welcome To ", "Our Project ", "We are proud ", "to have ", "Enam Sir."]

sen = 10  # more is less

while True:
    success, img = cap.read()
    imgText = np.zeros_like(img)
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]

        # cv2.circle(img,pointLeft,5,(255,0,255), cv2.FILLED)
        # cv2.circle(img,pointRight,5,(255,0,255), cv2.FILLED)
        # cv2.line(img, pointLeft, pointRight, (0, 200, 0), 3)

        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3
        f = 840
        d = (W * f) / w
        print(d)

        cvzone.putTextRect(img, f'Distance: {int(d)}cm', (face[10][0] - 125, face[10][1] - 65), scale=2)

    for i, text in enumerate(textList):
        singleHeight = 40 + int((int(d / sen) * sen) / 4)
        scale = 0.6 + (int(d / sen) * sen) / 80
        cv2.putText(imgText, text, (50, 50 + (i * singleHeight)), cv2.FONT_ITALIC, scale, (255, 255, 255), 2)
    imgStacked = cvzone.stackImages([img, imgText], 2, 1)

    cv2.imshow("Image", imgStacked)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
