# import cv2
# import numpy as np
#
# image = cv2.imread('images/photo.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image =cv2.resize(image,(600,800))
# kernel = np.ones((7,7),np.uint8)
# image = cv2.Canny(image,95,95)
#
# cv2.imshow('photo', image)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

image = cv2.imread('image/photo1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image =cv2.resize(image,(800,600))
kernel = np.ones((5,5),np.uint8)
image = cv2.Canny(image,400,400)

cv2.imshow('photo', image)

cv2.waitKey(0)
cv2.destroyAllWindows()