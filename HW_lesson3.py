import cv2
import numpy as np
image = cv2.imread('images/photo.jpg')

cv2.putText(image, 'Alexandra Tetiorkina', (60, 330), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
cv2.rectangle(image,(110, 70),(260, 310),(0, 0, 0),1)
cv2.imshow('photo', image)
cv2.waitKey(0)
cv2.destroyAllWindows()