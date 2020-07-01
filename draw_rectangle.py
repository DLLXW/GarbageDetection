import cv2
import os
cv_image=cv2.imread('1.jpg')
outs=[(156, 3,  976, 1445), (320, -21, 846, 1451)]
for c in outs:
    cv2.rectangle(cv_image, c,
                          (0, 0, 255), 4)
cv2.imwrite('1x.jpg', cv_image)
