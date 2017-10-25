""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np


black = (0, 0, 0)
white = (255, 255, 255)

cap = cv2.VideoCapture(0)

while True:

    # detect face
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20, 20))
    # create a NumPy matrix, which controls the degree of blurring
    kernel = np.ones((40, 40), 'uint8')

    for (x, y, w, h) in faces:
        # blurring the face
        frame[y:y+h, x:x+w, :] = cv2.dilate(frame[y:y+h, x:x+w, :], kernel)
        # draw a face
        # mouth
        cv2.line(frame, (int(x+w*.3), int(y+h*.75)), (int(x+w*.7), int(y+h*.75)), black, 10)
        # Nose
        cv2.line(frame, (int(x+w*.5), int(y+h*.55)), (int(x+w*.5), int(y+h*.4)), black, 10)
        # eyes
        cv2.circle(frame, (int(x+w*.7), int(y+h*.35)), 20, white, -1)
        cv2.circle(frame, (int(x+w*.3), int(y+h*.35)), 20, white, -1)
        cv2.circle(frame, (int(x+w*.7), int(y+h*.35)), 10, black, -1)
        cv2.circle(frame, (int(x+w*.3), int(y+h*.35)), 10, black, -1)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
