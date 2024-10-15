import cv2
print("[BİLGİ] Haar Cascade yüz tespit edici'yi yüklüyor...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
imgName = "IR_00135_RGB.png"
imgName, imgExtension = imgName.split('.')
img = cv2.imread(f"image/{imgName}.{imgExtension}")
# Haar Cascade gri tonlu resimler üzerinde çalıştığından renk uzayı dönüşümü yapalım
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # RGB uzaydan gri tonlu uzaya dönüşüm
print("[BİLGİ] Yüz tespiti gerçekleştiriliyor...")
scaleFactor, minNeighbors = 1.05, 25 # iki adet ayarlanabilir parametre
rects = detector.detectMultiScale(gray, scaleFactor, minNeighbors)
print(f"[BİLGİ] {len(rects)} adet yüz tespit edildi.")
# tespit edilen yüzleri resim üzerinde dikdörtgen olarak göster
for (x,y,w,h) in rects:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 9)
s = 0.22 # s=(0-1] resmi ekranda görüntülemek için ölçeklendir
rimg = cv2.resize(img, (int(s*img.shape[1]), int(s*img.shape[0])), 0)
cv2.imshow('Face detection with Haar Cascade', rimg)
cv2.waitKey(0)
cv2.imwrite(f"result/{imgName}_scaleFactor_{scaleFactor}_minNeighbors_{minNeighbors}.jpg", 
           img, [cv2.IMWRITE_JPEG_QUALITY, 50])