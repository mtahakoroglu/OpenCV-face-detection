import numpy as np
import cv2

# load our serialized model from disk
print("[INFO] loading model...")
# dnn deep neural network
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
conf = 0.5 # minimum probability to filter weak detections
# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
img_name, img_extension = "IR_00194_RGB_cropped", "png"
img = cv2.imread(f'image/{img_name}.{img_extension}')
(h, w) = img.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))

# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]

	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > conf:
		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		color = (0,255,0)
		cv2.rectangle(img, (startX, startY), (endX, endY), color, 5)
		cv2.putText(img, text, (startX+5, y-5), 0, 1, color, 3)
# save output image
cv2.imwrite(f"result/{img_name}-face-detection-DL.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])
# show the output image
s = 0.5
rimage = cv2.resize(img, (int(s*img.shape[1]), int(s*img.shape[0])), cv2.INTER_LINEAR)
cv2.imshow("Face detection with deep learning", rimage)
cv2.waitKey(0)