import numpy as np
import cv2

image = cv2.imread('image55.png')
image_blur = np.hstack(
    [
        cv2.blur(image,(3,3)),  #Mean Filter
        cv2.blur(image,(5,5)),
        cv2.blur(image,(7,7))   # we can see the picture become more dim
    ]
)
cv2.imwrite('blur_of_diff_mean_size.jpg',image_blur)

image_gaussian = np.hstack([
    cv2.GaussianBlur(image, (3, 3), 0), # 0 means antomative calculate the weight of every pix
    cv2.GaussianBlur(image, (5, 5), 0),
    cv2.GaussianBlur(image, (7, 7), 0)   # not so dim like blur
])
cv2.imwrite('blur_of_diff_gaussian_size.jpg', image_gaussian)

image_mid_blur = np.hstack([
    cv2.medianBlur(image, 3),   #for salt-peper noisy picture using median Filter
    cv2.medianBlur(image, 5),
    cv2.medianBlur(image, 7)    # 邻域越大，过滤椒盐噪声效果越好，但是图像质量也会下降明显。除非非常密集椒盐噪声，否则不推荐Ksize=7这么大的卷积核
])
cv2.imwrite('blur_of_diff_median_size.jpg', image_mid_blur)
#cv2.bilaterFilter(image, Ksize, sigmaColor, sigmaSpace)
image_bilater = np.hstack([     #seemingly like double gaussian Filter.
    cv2.bilateralFilter(image, 5, 21, 21),
    cv2.bilateralFilter(image, 7, 31, 31),
    cv2.bilateralFilter(image, 9, 41, 41)
])
cv2.imwrite('blur_of_diff_bilater_size.jpg', image_bilater)

image_summary = np.vstack([
    image_blur,
    image_gaussian,
    image_mid_blur,
    image_bilater
])
cv2.imwrite('Summary.jpg',image_summary)
"""
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
k=cv2.waitKey(0)&0xFF
if(k==27):
    cv2.destroyAllWindows()
elif(k==ord('s')):
    cv2.imwrite('Result.jpg',img)
    cv2.destroyAllWindows()
"""
