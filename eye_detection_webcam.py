import cv2

vid = cv2.VideoCapture(0)
eye_cascade = cv2.CascadeClassifier("C:\\Open CV\\haar_cascade\\eye.xml")

while 1:
    _ , frame = vid.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 9)
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("cagan",frame)

    if cv2.waitKey(15) & 0xFF==ord("q"):
        break
vid.release()
cv2.destroyAllWindows()