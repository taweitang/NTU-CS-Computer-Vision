import cv2
import numpy as np

def UpsideDown (a_img) :

    img = np.zeros((a_img.shape[0], a_img.shape[1],a_img.shape[2]),dtype=np.uint8)

    for i in range (a_img.shape [0]):
        for j in range (a_img.shape[1]):
            for k in range (a_img.shape[2]):

                img[i,j,k] = a_img[a_img.shape[0]-i-1,j,k]

    return img

def RightSideLeft (a_img) :

    img = np.zeros((a_img.shape[0], a_img.shape[1],a_img.shape[2]),dtype=np.uint8)

    for i in range (a_img.shape [0]):
        for j in range (a_img.shape[1]):
            for k in range (a_img.shape[2]):

                img[i,j,k] = a_img[i,a_img.shape[1]-j-1,k]

    return img

def DiagonallyMirrored (a_img) :

    img = np.zeros((a_img.shape[0], a_img.shape[1],a_img.shape[2]),dtype=np.uint8)

    for i in range (a_img.shape [0]):
        for j in range (a_img.shape[1]):
            for k in range (a_img.shape[2]):

                img[i,j,k] = a_img[a_img.shape[0]-i-1,a_img.shape[1]-j-1,k]

    return img

# load and process
m_lena = cv2.imread("C:\\Users\\USER\\Desktop\\lena.bmp")
m_upsideDown = UpsideDown(m_lena)
m_rightSideLeft = RightSideLeft(m_lena)
m_diagonallyMirrored = DiagonallyMirrored(m_lena)

# save
cv2.imwrite('upsideDown.bmp', m_upsideDown)
cv2.imwrite('rightSideLeft.bmp', m_rightSideLeft)
cv2.imwrite('diagonallyMirrored.bmp', m_diagonallyMirrored)

# show
cv2.imshow("lena",m_lena)
cv2.imshow("upside-down",m_upsideDown)
cv2.imshow("right-side-left",m_rightSideLeft)
cv2.imshow("diagonally mirrored",m_diagonallyMirrored)
cv2.waitKey(0)
cv2.destroyAllWindows()