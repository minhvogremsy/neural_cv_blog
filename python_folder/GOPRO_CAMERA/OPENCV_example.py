import cv2 
import time
#?overrun_nonfatal=1&fifo_size=50000000
cap = cv2.VideoCapture('udp://@172.29.161.53:8554?overrun_nonfatal=1&fifo_size=50000000', cv2.CAP_DSHOW)
if not cap.isOpened():
    print('VideoCapture not opened')
    exit(-1)
print(1)

while True:

    timer = cv2.getTickCount()
    time.sleep(0.05)
    ret, frame = cap.read()
    #print(frame.shape)
    if not ret:
        print('frame empty')
        break

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    if fps > 60:
                                myColor = (20, 230, 20)
    elif fps > 20:
                                myColor = (230, 20, 20)
    else:
                                myColor = (20, 20, 230)


    cv2.putText(frame, str(int(fps)), (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, myColor, 2)
    cv2.imshow('image', frame)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()