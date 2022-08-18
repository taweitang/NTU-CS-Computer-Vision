import cv2
import numpy as np
import matplotlib.pyplot as plt

def Divided  (a_img,a_divisor) :

    h,w = a_img.shape[0],a_img.shape[1]
    img = np.zeros((h, w,1),dtype=np.uint8)

    for i in range (h):
        for j in range (w):   
                img[i,j,0] = int(a_img[i,j,0]/3)

    return img

def Histogram (a_img) :

    h,w = a_img.shape[0],a_img.shape[1]
    histogram = np.zeros(256)

    for i in range (h):
        for j in range (w):       
            gray = int(a_img[i,j,0])
            histogram[gray] += 1

    return histogram

def Equalization (a_img) :

    h,w = a_img.shape[0],a_img.shape[1]
    histogram = Histogram(a_img)
    eqarray = np.zeros(256)
    img = np.zeros((h, w,1),dtype=np.uint8)

    for i in range(256):
        for j in range(i):
            eqarray[i] += histogram[j]/(h*w)
        eqarray[i]*=255
        eqarray[i] = int(eqarray[i])

    for i in range (h):
        for j in range (w):   
            index = int(a_img[i,j,0])
            img[i,j,0] = eqarray[index]

    return img



# Write a program to generate images and histograms:

# (a) original image and its histogram
m_lena = cv2.imread("lena.bmp")
m_Histogram = Histogram(m_lena)
plt.bar(range(1,257), m_Histogram)
plt.savefig("Histogram.jpg")
# (b) image with intensity divided by 3 and its histogram
m_Divided = Divided(m_lena,3)
cv2.imwrite("Divided.jpg",m_Divided)
m_DividedHistogram = Histogram(m_Divided)
plt.figure()
plt.bar(range(1,257), m_DividedHistogram)
plt.savefig("DividedHistogram.jpg")
# (c) image after applying histogram equalization to (b) and its histogram
m_Equalization = Equalization(m_Divided)
cv2.imwrite("Equalization.jpg",m_Equalization)
plt.figure()
m_EqualizationHistogram = Histogram(m_Equalization)
plt.bar(range(1,257), m_EqualizationHistogram)
plt.savefig("EqualizationHistogram.jpg")