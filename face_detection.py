import cv2

img = cv2.imread("C:\\Open CV\\test_images\\ecem.png")
face_cascade = cv2.CascadeClassifier("C:\\Open CV\\haar_cascade\\frontalface.xml")

## convert rgb to gray scale

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## the following function detects the faces in the image and store these as tuple in the variable faces
faces= face_cascade.detectMultiScale(gray, 1.3 , 4)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)



cv2.imshow('image',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
