import cv2
import numpy as np


m_kernel = np.array([[0,1,1,1,0],
                    [1,1,1,1,1],
                    [1,1,1,1,1],
                    [1,1,1,1,1],
                    [0,1,1,1,0]])

m_kernel_j = np.array([[0,0,0,0,0],
                    [0,0,0,0,0],
                    [1,1,0,0,0],
                    [0,1,0,0,0],
                    [0,0,0,0,0]])

m_kernel_k = np.array([[0,0,0,0,0],
                    [0,1,1,0,0],
                    [0,0,1,0,0],
                    [0,0,0,0,0],
                    [0,0,0,0,0]])


def Binarize  (a_img,a_threshold) :

    img = np.zeros((a_img.shape[0], a_img.shape[1]),np.int)

    for i in range (a_img.shape [0]):
        for j in range (a_img.shape[1]):   
            if (a_img[i,j,0] > a_threshold - 1):

                img[i,j] = 255

    return img

def Dilation (a_img,ker) :

    h,w = a_img.shape[0],a_img.shape[1]

    ker_h, ker_w = ker.shape[0], ker.shape[1]

    row_d, column_d = int((ker_h - 1) / 2), int((ker_w - 1) / 2)

    temp_img = np.zeros((h + 2 * row_d, w + 2 * column_d), np.int)

    temp_img[row_d:h + row_d, column_d:w + column_d] = a_img

    dilation = np.zeros((h, w), np.int)

    for i in range(row_d, h + row_d):
        for j in range(column_d, w + column_d):
            if np.any(np.logical_and(ker,temp_img[i - row_d: i + row_d + 1, j - column_d: j + column_d + 1])):
                dilation[i - row_d, j - column_d] = 255

    return dilation

def Erosion (a_img,ker) :

    h,w = a_img.shape[0],a_img.shape[1]

    ker_h, ker_w = ker.shape[0], ker.shape[1]

    row_d, column_d = int((ker_h - 1) / 2), int((ker_w - 1) / 2)

    temp_img = np.zeros((h + 2 * row_d, w + 2 * column_d), np.int)

    temp_img[row_d:h + row_d, column_d:w + column_d] = a_img

    erosion = np.zeros((h, w), np.int)

    for i in range(row_d, h + row_d):
        for j in range(column_d, w + column_d):
            if not np.any(ker - np.logical_and(ker, temp_img[i - row_d: i + row_d + 1,j - column_d: j + column_d + 1])):
                erosion[i - row_d, j - column_d] = 255

    return erosion

def Opening (a_img,ker):

    erosion = Erosion(a_img,ker)
    opening = Dilation(erosion,ker)

    return opening

def Closing (a_img,ker):

    dilation = Dilation(a_img,ker)
    closing = Erosion(dilation,ker)

    return closing

def Binary_image_complement(a_img):
    h,w = a_img.shape[0],a_img.shape[1]
    new_img = np.zeros((h, w), np.int)

    for i in range(h):
        for j in range(w):
            new_img[i, j] = 255 - a_img[i, j]

    return new_img


def Hit_and_miss(a_img, ker_j, ker_k):

    h,w = a_img.shape[0],a_img.shape[1]
    hit_and_miss = np.zeros((h, w), np.int)
    temp_img1 = Erosion(a_img, ker_j)
    temp_img2 = Erosion(Binary_image_complement(a_img), ker_k)
    for i in range(h):
        for j in range(w):
            if temp_img1[i, j] == 255 and temp_img2[i, j] == 255:
                hit_and_miss[i, j] = 255


    return hit_and_miss



# Write a program to generate images and histograms:

# (a) Dilation
m_lena = cv2.imread("lena.bmp")
m_Binarize = Binarize(m_lena,128)
m_Dilation = Dilation(m_Binarize,m_kernel)
cv2.imwrite("Dilation.jpg",m_Dilation)
print("Dilation done")
# (b) Erosion
m_Erosion = Erosion(m_Binarize,m_kernel)
cv2.imwrite("Erosion.jpg",m_Erosion)
print("Erosion done")
# (c) Opening
m_Opening = Opening(m_Binarize,m_kernel)
cv2.imwrite("Opening.jpg",m_Opening)
print("Opening done")
# (d) Closing
m_Closing = Closing(m_Binarize,m_kernel)
cv2.imwrite("Closing.jpg",m_Closing)
print("Closing done")
# (e) Hit-and-miss transform
m_Hit_and_miss = Hit_and_miss(m_Binarize,m_kernel_j,m_kernel_k)
cv2.imwrite("Hit_and_miss.jpg",m_Hit_and_miss)
print("Hit_and_miss done")


