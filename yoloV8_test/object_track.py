import cv2

tracker = cv2.TrackerCSRT_create()
tracking = False

cap = cv2.VideoCapture("C:\yolov8_awsRekognition\C0896.mp4")
if not cap.isOpened():
    raise Exception("影片無法開啟")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    # frame = cv2.flip(frame, 1)
    # frame = cv2.resize(frame, (540, 300))
    keyName = cv2.waitKey(1)

    if keyName == ord('q'):
        break
    if keyName == ord('a'):
        area = cv2.selectROI('track', frame,
                             showCrosshair=False, fromCenter=False)
        tracker.init(frame, area)
        tracking = True
    if tracking:
        success, point = tracker.update(frame)
        if success:
            p1 = [int(point[0]), int(point[1])]
            p2 = [int(point[0] + point[2]), int(point[1] + point[3])]
            cv2.rectangle(frame, p1, p2, (0, 0, 255),
                          3)

    cv2.imshow('track', frame)

cap.release()
cv2.destroyAllWindows()
