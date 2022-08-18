import cv2
import numpy as np

original_img = cv2.imread('lena.bmp', 0)

def robert(img,threshold):
    ker1 = np.array([[-1,0],[0,1]])
    ker2 = np.array([[0,-1],[1,0]])

    row,col = img.shape

    temp_img = cv2.copyMakeBorder(img,0,1,0,1,cv2.BORDER_CONSTANT,value=0)
    new_img = np.zeros((row,col))

    for i in range (row):
        for j in range (col):
            temp = temp_img[i:i+2,j:j+2]
            new_img[i,j] = np.sqrt(np.sum(np.multiply(temp,ker1))**2 + np.sum(np.multiply(temp,ker2))**2)

    return_img = np.zeros((row,col))
    for i in range (row):
        for j in range (col):
            if (new_img[i,j]<threshold):
                return_img[i,j] = 255

    return return_img

def perwitt(img,threshold):
    ker1 = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    ker2 = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

    row,col = img.shape

    temp_img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
    new_img = np.zeros((row,col))

    for i in range (row):
        for j in range (col):
            temp = temp_img[i:i+3,j:j+3]
            new_img[i,j] = np.sqrt(np.sum(np.multiply(temp,ker1))**2 + np.sum(np.multiply(temp,ker2))**2)

    return_img = np.zeros((row,col))
    for i in range (row):
        for j in range (col):
            if (new_img[i,j]<threshold):
                return_img[i,j] = 255

    return return_img

def sobel(img,threshold):
    ker1 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    ker2 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    row,col = img.shape

    temp_img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
    new_img = np.zeros((row,col))

    for i in range (row):
        for j in range (col):
            temp = temp_img[i:i+3,j:j+3]
            new_img[i,j] = np.sqrt(np.sum(np.multiply(temp,ker1))**2 + np.sum(np.multiply(temp,ker2))**2)

    return_img = np.zeros((row,col))
    for i in range (row):
        for j in range (col):
            if (new_img[i,j]<threshold):
                return_img[i,j] = 255

    return return_img

def frei_chen(img,threshold):
    ker1 = np.array([[-1,-np.sqrt(2),-1], [0,0,0], [1,np.sqrt(2),1]])
    ker2 =  np.array([[-1,0,1], [-np.sqrt(2),0,np.sqrt(2)], [-1,0,1]])

    row,col = img.shape

    temp_img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
    new_img = np.zeros((row,col))

    for i in range (row):
        for j in range (col):
            temp = temp_img[i:i+3,j:j+3]
            new_img[i,j] = np.sqrt(np.sum(np.multiply(temp,ker1))**2 + np.sum(np.multiply(temp,ker2))**2)

    return_img = np.zeros((row,col))
    for i in range (row):
        for j in range (col):
            if (new_img[i,j]<threshold):
                return_img[i,j] = 255

    return return_img

def krisch(img,threshold):
    ker_k0 = np.array([[-3,-3,5], [-3,0,5], [-3,-3,5]])
    ker_k1 = np.array([[-3,5,5], [-3,0,5], [-3,-3,-3]])
    ker_k2 = np.array([[5,5,5], [-3,0,-3], [-3,-3,-3]])
    ker_k3 = np.array([[5,5,-3], [5,0,-3], [-3,-3,-3]])
    ker_k4 = np.array([[5,-3,-3], [5,0,-3], [5,-3,-3]])
    ker_k5 = np.array([[-3,-3,-3], [5,0,-3], [5,5,-3]])
    ker_k6 = np.array([[-3,-3,-3], [-3,0,-3], [5,5,5]])
    ker_k7 = np.array([[-3,-3,-3], [-3,0,5], [-3,5,5]])
    list_kn = [ker_k0, ker_k1, ker_k2, ker_k3, ker_k4, ker_k5, ker_k6, ker_k7]
    rows, cols = img.shape
    temp_img = cv2.copyMakeBorder(src=img, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_REPLICATE)
    new_img = img.copy().astype(float)
    for i in range(rows):
        for j in range(cols):
            temp = temp_img[i:i+3, j:j+3]
            max=0 # 初始化max值
            for ker in list_kn:
                temp_sum = np.sum(ker * temp)
                if temp_sum > max:
                    max = temp_sum
            new_img[i, j] = max

    return_img = np.zeros((rows,cols))
    for i in range (rows):
        for j in range (cols):
            if (new_img[i,j]<threshold):
                return_img[i,j] = 255

    return return_img

def robinson(img,threshold):
    ker_r0 = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    ker_r1 = np.array([[0,1,2], [-1,0,1], [-2,-1,0]])
    ker_r2 = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
    ker_r3 = np.array([[2,1,0], [1,0,-1], [0,-1,-2]])
    ker_r4 = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
    ker_r5 = np.array([[0,-1,-2], [1,0,-1], [2,1,0]])
    ker_r6 = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
    ker_r7 = np.array([[-2,-1,0], [-1,0,1], [0,1,2]])
    list_rn = [ker_r0, ker_r1, ker_r2, ker_r3, ker_r4, ker_r5, ker_r6, ker_r7]
    rows, cols = img.shape
    temp_img = cv2.copyMakeBorder(src=img, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_REPLICATE)
    new_img = img.copy().astype(float)
    for i in range(rows):
        for j in range(cols):
            temp = temp_img[i:i+3, j:j+3]
            max=0 # 初始化max值
            for ker in list_rn:
                temp_sum = np.sum(ker * temp)
                if temp_sum > max:
                    max = temp_sum
            new_img[i, j] = max
    return_img = np.zeros((rows,cols))
    for i in range (rows):
        for j in range (cols):
            if (new_img[i,j]<threshold):
                return_img[i,j] = 255

    return return_img


def nevatia_babu(img,threshold):
    ker_nb0 = np.array([[100,100,100,100,100], [100,100,100,100,100], [0,0,0,0,0], [-100,-100,-100,-100,-100], [-100,-100,-100,-100,-100]])
    ker_nb1 = np.array([[100,100,100,100,100], [100,100,100,78,-32], [100,92,0,-92,-100], [32,-78,-100,-100,-100], [-100,-100,-100,-100,-100]])
    ker_nb2 = np.array([[100,100,100,32,-100], [100,100,92,-78,-100], [100,100,0,-100,-100], [100,78,-92,-100,-100], [100,-32,-100,-100,-100]])
    ker_nb3 = np.array([[-100,-100,0,100,100], [-100,-100,0,100,100], [-100,-100,0,100,100], [-100,-100,0,100,100], [-100,-100,0,100,100]])
    ker_nb4 = np.array([[-100,32,100,100,100], [-100,-78,92,100,100], [-100,-100,0,100,100], [-100,-100,-92,78,100], [-100,-100,-100,-32,100]])
    ker_nb5 = np.array([[100,100,100,100,100], [-32,78,100,100,100], [-100,-92,0,92,100], [-100,-100,-100,-78,32], [-100,-100,-100,-100,-100]])

    list_nbn = [ker_nb0, ker_nb1, ker_nb2, ker_nb3, ker_nb4, ker_nb5]
    rows, cols = img.shape
    temp_img = cv2.copyMakeBorder(src=img, top=2, bottom=2, left=2, right=2, borderType=cv2.BORDER_REPLICATE)
    new_img = img.copy().astype(float)
    for i in range(rows):
        for j in range(cols):
            temp = temp_img[i:i+5, j:j+5]
            max=0 # 初始化max值
            for ker in list_nbn:
                temp_sum = np.sum(ker * temp)
                if temp_sum > max:
                    max = temp_sum
            new_img[i, j] = max
    return_img = np.zeros((rows,cols))
    for i in range (rows):
        for j in range (cols):
            if (new_img[i,j]<threshold):
                return_img[i,j] = 255

    return return_img





robert_img = robert(original_img,12)
cv2.imwrite('robert_img.bmp',robert_img)
perwitt_img = perwitt(original_img,24)
cv2.imwrite('perwitt_img.bmp',perwitt_img)
sobel_img = sobel(original_img,38)
cv2.imwrite('sobel_img.bmp',sobel_img)
frei_chen_img = frei_chen(original_img,30)
cv2.imwrite('frei_chen_img.bmp',frei_chen_img)
krisch_img = krisch(original_img,135)
cv2.imwrite('krisch_img.bmp',krisch_img)
robinson_img = robinson(original_img,43)
cv2.imwrite('robinson_img.bmp',robinson_img)
nevatia_babu_img = nevatia_babu(original_img,12500)
cv2.imwrite('nevatia_babu_img.bmp',nevatia_babu_img)


    