import cv2 as cv
print("[BİLGİ] Haar Cascade yüz tespit edici'yi yüklüyor...")
detector = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
imgName, imgExtension = "IMG_20220824_111044", "jpg"
img = cv.imread(f"image/{imgName}.{imgExtension}")
# Haar Cascade gri tonlu resimler üzerinde çalıştığından renk uzay dönüşümü yapalım
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print("[BİLGİ] Yüz tespiti gerçekleştiriliyor...")
scaleFactor, minNeighbors = 1.05, 35
rects = detector.detectMultiScale(gray, scaleFactor, minNeighbors)
print(f"[BİLGİ] {len(rects)} adet yüz tespit edildi.")
for (x,y,w,h) in rects:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 9)
s = 0.25 # resmi ekranda görüntülemek için ölçeklendir
rimg = cv.resize(img, (int(s*img.shape[1]), int(s*img.shape[0])), 0)
cv.imshow('Face detection with Haar Cascade', rimg)
cv.waitKey(0)
cv.imwrite(f"{imgName}_scaleFactor_{scaleFactor}_minNeighbors_{minNeighbors}.jpg", 
           img, [cv.IMWRITE_JPEG_QUALITY, 50])