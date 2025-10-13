import cv2
import numpy as np

img = cv2.imread('images/kr.jpg')
img_copy = img.copy()
img = cv2.GaussianBlur(img,(5,5),2)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([153,0,0])
upper_red = np.array([179,255,255])
mask_red = cv2.inRange(img, lower_red, upper_red)

lower_green = np.array([7,0,24])
upper_green = np.array([86,255,255])
mask_green = cv2.inRange(img, lower_green, upper_green)

lower_blue = np.array([105,163,104])
upper_blue = np.array([134,255,255])
mask_blue = cv2.inRange(img, lower_blue, upper_blue)

lower_yellow = np.array([7,0,24])
upper_yellow = np.array([86,255,255])
mask_yellow = cv2.inRange(img, lower_yellow, upper_yellow)

mask_total = cv2.bitwise_or(mask_red, mask_blue)
mask_total = cv2.bitwise_or(mask_total, mask_green)
mask_total = cv2.bitwise_or(mask_total, mask_yellow)

contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 10:
        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])


        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = round(w / h, 2)
        compactness = round((4 * np.pi * area) / (perimeter ** 2), 2)

        approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, True)

        for h in hsw_to_rgb(cnt):
            if h < 10 or h > 160 and h < 179:
                a = "red"
            elif h > 36 and h < 85:
                a = "green"
            elif h > 86 and h < 100:
                a = "blue"
            elif h > 26 and h < 35:
                a = "green"
            else:
                a = 'other'

        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            shape = "Square"
        elif len(approx) > 10:
            shape = "Oval"
        else:
            shape = "Other"

        cv2.drawContours(img_copy, [cnt], -1, (0, 255, 0), 2)
        cv2.circle(img_copy, (cx, cy), 4, (0, 255, 0), 1)
        cv2.putText(img_copy, f'Shape: {shape}', (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (0, 0, 255), 1)
        cv2.putText(img_copy, f'Area: {area}, P: {int(perimeter)}, Color: {a}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (0, 0, 255), 1)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_copy, f'Aspect ratio: {aspect_ratio}, compactness: {compactness}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (0, 0, 255), 1)


cv2.imshow('image',img)
cv2.imshow('mask',img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()