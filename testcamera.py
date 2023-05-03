import cv2 
cap = cv2.VideoCapture(2)

while True:
    ret, frame =  cap.read()
    if ret == False:
        print("Something went wrong!")
        break
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
