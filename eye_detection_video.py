import cv2

vid = cv2.VideoCapture("C:\\Open CV\\test_videos\\WhatsApp Video 2022-11-04 at 20.30.06.mp4")
face_cascade = cv2.CascadeClassifier("C:\\Open CV\\haar_cascade\\frontalface.xml")
eye_cascade = cv2.CascadeClassifier("C:\\Open CV\\haar_cascade\\eye.xml")

while True:
    _ , frame= vid.read()
    frame = cv2.resize(frame,(480,360))
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    frame2 = frame[y:y+h,x:x+w]
    roi_gray = gray[y:y+h,x:x+w]

    eyes=eye_cascade.detectMultiSclae(roi_gray,1.3,5)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame2, (ex, ey), (ex + ew, ey + eh), (0, 135, 255), 2)

    if cv2.waitKey(15) & 0xFF==ord("q"):
        break
vid.release()
cv2.destroyAllWindows()
