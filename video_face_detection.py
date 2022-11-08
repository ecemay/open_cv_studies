import cv2

vid = cv2.VideoCapture("C:\\Open CV\\test_videos\\WhatsApp Video 2022-11-04 at 20.30.06.mp4")
face_cascade = cv2.CascadeClassifier("C:\\Open CV\\haar_cascade\\frontalface.xml")

while 1:
    _ , frame= vid.read()
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 9)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("cagan",frame)

    if cv2.waitKey(15) & 0xFF==ord("q"):
        break
vid.release()
cv2.destroyAllWindows()