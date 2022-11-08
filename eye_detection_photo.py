import cv2

img = cv2.imread("C:\\Open CV\\test_images\\ecem.png")
face_cascade = cv2.CascadeClassifier("C:\\Open CV\\haar_cascade\\frontalface.xml")
eye_cascade = cv2.CascadeClassifier("C:\\Open CV\\haar_cascade\\eye.xml")

## convert rgb to gray scale

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## the following function detects the faces in the image and store these as tuple in the variable faces
faces= face_cascade.detectMultiScale(gray, 1.3 , 6)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

img2 = img[y:y+h,x:x+w]
gray2 = gray[y:y+h,x:x+w]

eyes= eye_cascade.detectMultiScale(gray2)
for (ex,ey,ew,eh) in eyes:
    cv2.rectangle(img2,(ex,ey),(ex+ew,ey+eh),(0,135,255),2)

cv2.imshow("image",img2)

cv2.waitKey(0)
cv2.destroyAllWindows()


