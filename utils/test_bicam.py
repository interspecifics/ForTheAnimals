import cv2
from PIL import Image

cap = cv2.VideoCapture(1)
isA = True

cv2_ima = None
cv2_imb = None

while True:
    if (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame
        cv2.imshow('frame', cv2_im)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('l'):
        cap.release()
        cap = cv2.VideoCapture(0)
        isA = True
    elif key & 0xFF == ord('r'):
        cap.release()
        cap = cv2.VideoCapture(4)
        isA = False
    if key & 0xFF == ord('q'):
        cap.release()
        break
cv2.destroyAllWindows()


