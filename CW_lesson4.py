import cv2
import numpy as np
img = np.full((400, 600, 3), (203, 192, 255), dtype=np.uint8) # для заливки фону
cv2.rectangle(img,(5, 5),(595, 395),(232, 161, 116),2)
photo = cv2.imread("images/photo.jpg")
photo_small = cv2.resize(photo, (120, 150))
img[10:160, 10:130] = photo_small
qr = cv2.imread("images/qr1.png")
qr_small = cv2.resize(qr, (140, 140))
h, w = qr_small.shape[:2]
img[-h-10:-10, -w-10:-10] = qr_small

cv2.putText(img, 'Alexandra Tetiorkina', (145, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
cv2.putText(img, 'Computer Vision Student', (145, 110), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 1)
cv2.putText(img, 'Email: atrobber@gmail.com', (145, 200), cv2.FONT_HERSHEY_DUPLEX, 0.75, (232, 161, 116), 1)
cv2.putText(img, 'Phone: +380 66 539 6022', (145, 230), cv2.FONT_HERSHEY_DUPLEX, 0.75, (232, 161, 116), 1)
cv2.putText(img, 'Date: 08/03/2010', (145, 260), cv2.FONT_HERSHEY_DUPLEX, 0.75, (232, 161, 116), 1)
cv2.putText(img, 'OpenCV Buisness Card', (145, 360), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 2)

cv2.imshow("buisness_card", img)
cv2.imwrite("business_card.png", img)



cv2.waitKey(0)
cv2.destroyAllWindows()