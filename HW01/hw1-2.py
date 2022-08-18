import cv2
import numpy as np

def Rotate (a_img,angle) :

    (h, w) = a_img.shape[:2]
    center = (w / 2, h / 2)
    img = cv2.getRotationMatrix2D(center, -angle, 1.0)
    img = cv2.warpAffine(a_img, img, (w, h))

    return img

def Shrink (a_img,ratio) :

    (h, w) = a_img.shape[:2]
    img = cv2.resize(a_img, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_CUBIC)

    return img

def Binarize (a_img,threashhold) :

    img = cv2.threshold(a_img,threashhold,255,cv2.THRESH_BINARY)[1]

    return img

# load and process
m_lena = cv2.imread("C:\\Users\\USER\\Desktop\\lena.bmp")
m_lena_gray = cv2.imread("C:\\Users\\USER\\Desktop\\lena.bmp",0)
m_rotate = Rotate(m_lena,45)
m_shrink = Shrink(m_lena,0.5)
m_binarize = Binarize(m_lena_gray,128)

# save
cv2.imwrite('Rotate.bmp', m_rotate)
cv2.imwrite('Shrink.bmp', m_shrink)
cv2.imwrite('Binarize.bmp', m_binarize)

# show
cv2.imshow("lena",m_lena)
cv2.imshow("rotate",m_rotate)
cv2.imshow("shrink",m_shrink)
cv2.imshow("binarize",m_binarize)
cv2.waitKey(0)
cv2.destroyAllWindows()