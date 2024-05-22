import cv2
import time
from collections import deque
import numpy as np
print('[INFO] loading haar cascade face detector...')
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
previousTime = time.time()
fps = deque(maxlen=100)
showFPS, k = True, 0 # decide if you like to show fps count or not, frame number
while True:
    ret, frame = cap.read()
    k = k+1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    currentTime = time.time()
    fpsCurrent = 1 / (currentTime-previousTime)
    fps.append(fpsCurrent)
    fpsAvg = np.mean(fps)
    if showFPS:
        cv2.putText(frame, f"fps = {fpsAvg:.2f}", (30,50), 0, 1, (0,0,0), 2)
    cv2.imshow('Web cam stream', frame)
    previousTime = currentTime
    if cv2.waitKey(1) & 0xFF == ord('c'): # kullanıcı c tuşuna basarsa görüntüyü kaydet
        cv2.imwrite(f"result/haar_cascade_web_cam_frame_{k}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(f"Frame #{k} is saved to hard disk.")
    elif cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()