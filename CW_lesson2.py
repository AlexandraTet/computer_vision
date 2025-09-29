import cv2
import numpy
import numpy as np

# image = cv2.imread('images/sample.jpg')
# # image = cv2.resize(image,(400,600))
# print(image.shape)
# image = cv2.resize(image,(image.shape[1] // 2, image.shape[0] // 2))
# # image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
# # image = cv2.flip(image, 1)
# # image = cv2.GaussianBlur(image,(7,7),5) # рівень блюру має бути лише непарних чисел
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.Canny(image,100,100)
# # image = cv2.dilate(image,None,iterations=1) # обводка кольору, матриця значень для опрацювання і обведення (весь малюнок величезний список при none)
# kernel = np.ones((5,5),numpy.uint8)
# image = cv2.dilate(image,kernel,iterations = 1)
# image = cv2.erode(image,kernel,iterations = 1)
# cv2.imshow('frog', image)
# # cv2.imshow( 'image', image[0:200, 0:400]) # обрізати фрагмент

# VIDEO
# video = cv2.VideoCapture('video/vid1.mp4')
video = cv2.VideoCapture(0)
while True:
    mistake, frame = video.read()
    frame = cv2.resize(frame,(800,600))
    cv2.imshow('video',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cv2.waitKey(0) #час показу зображення. 0 = постійне, до закриття.
cv2.destroyAllWindows()