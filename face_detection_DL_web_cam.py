import numpy as np
import time
import cv2
from collections import deque

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
conf = 0.5 # minimum probability to filter weak detections
cap = cv2.VideoCapture(0)
fps = deque(maxlen=100)
k = 0 # index for saving screenshot to file
previousTime = time.time() # capture timestamp to initialize fps computation
showFPS, showFrameNumber = True, True # decide if you like to show fps count or not, frame number
while True:
	ret, frame = cap.read()
	k = k+1
	# grab the frame dimensions and convert it to a blob
	(h,w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
	# pass the blob through the network and obtain the detections and predictions
	net.setInput(blob)
	detections = net.forward()
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		confidence = detections[0, 0, i, 2]
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < conf:
			continue
		# compute the (x,y)-coordinates of the bounding box for the object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		# draw the bounding box of the face along with the associated probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX,startY), (endX,endY), (0, 0, 255), 2)
		cv2.putText(frame, text, (startX,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	currentTime = time.time()
	fpsCurrent = 1 / (currentTime - previousTime)
	fps.append(fpsCurrent)
	fpsAvg = np.mean(fps)
	if showFPS:
		cv2.putText(frame, f"fps = {fpsAvg:.2f}", (30,50), 0, 1, (0,0,0), 2)
	if showFrameNumber:
		cv2.putText(frame, f"frame #{k}", (frame.shape[1]-215,50), 0, 1, (0,0,0), 2)
	cv2.imshow("face detection with deep learning", frame)
	previousTime = currentTime
	if cv2.waitKey(1) & 0xFF == ord('c'): # kullanıcı c tuşuna basarsa görüntüyü kaydet
		cv2.imwrite(f"result/deep_learning_face_detection_frame_{k}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
		print(f"Frame #{k} is saved to hard disk.")
	elif cv2.waitKey(1) == 27:
		break
cap.release()
cv2.destroyAllWindows()