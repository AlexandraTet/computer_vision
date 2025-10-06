import cv2
import numpy as np
img = cv2.imread('images/face.jpg')
scale = 1
img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
print(img.shape)


img_copy = img.copy()
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
img_copy = cv2.GaussianBlur(img_copy, (5, 5), 2) # blur image
img_copy_color = img.copy()
img_copy = cv2.equalizeHist(img_copy) # посилення контрасту

img_copy = cv2.Canny(img_copy, 50, 170)

# border searching. RETR_EXTERNAL - режим отримання контурів, знаходить крайні контури, якщо об`єкт має дирку вона буде ігноруватися. CHAIN_APPROX - апроксимація, процес наближенного вираження одних величин або об`єктів через інші
contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#  малювання контурів прямокутників та тексту


for cnt in contours:
    area = cv2.contourArea(cnt) # визначення площі контору, повертає результат в пікселях
    if area > 120:
        x, y, w, h = cv2.boundingRect(cnt) # дозволяє створити найменший прямокутник, який містить в собі цей контур

        cv2.drawContours(img_copy_color, [cnt], -1, (0, 255, 0), 2)

        cv2.rectangle(img_copy_color, (x, y), (x + w, y + h), (255, 255, 255), 2)
        text_y = y - 5 if y - 5 > 10 else y + 15
        text = f'x: {x}, y: {y}, S: {int(area)}'
        cv2.putText(img_copy_color, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)



cv2.imshow('Copy border', img_copy)
cv2.imshow('Copy border', img_copy_color)
cv2.imshow('Borders', img)
cv2.waitKey(0)
cv2.destroyAllWindows()