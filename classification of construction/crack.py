import numpy as np
import cv2
import matplotlib.pyplot as plt

def onChange():
    pass

image = cv2.imread("crack_image/big (1).jpg", cv2.IMREAD_GRAYSCALE) # 영상 읽기
if image is None: raise Exception("영상파일 읽기 오류")

blur = cv2.GaussianBlur(image,(0,0),1)

fig = plt.figure()

ax = fig.add_subplot(111)

th1 = 200
th2 = 600

skeleton = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                                 cv2.THRESH_BINARY, 7, -3)

img = cv2.Canny(blur, th1, th2)
# img = img[:,3]
# ax.imshow(image)
# ax.imshow(skeleton)
# cv2.circle(image, img, 3, (0,0,255), -1)
# ax.imshow(img, alpha=0.4)

cv2.imshow('',img)
cv2.waitKey(0)

plt.savefig(('add.jpg'), dpi=500)
plt.close('all')

# cv2.imshow("ca",img)

# # cv2.imshow("d",image)
# cv2.waitKey(0)

# cv2.namedWindow("result")
# a = cv2.getTrackbarPos('A', 'result')
# cv2.createTrackbar('A', 'result', 0, 255, a)

# cv2.createTrackbar('B', 'result', 0, 255, onChange)
# b = cv2.getTrackbarPos('B', 'result')
# while(True):
#     canny_img = cv2.Canny(image, a, b)

#     cv2.imshow("canny_img", canny_img)
#     cv2.waitKey(0)