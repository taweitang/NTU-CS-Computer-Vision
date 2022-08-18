import cv2
import numpy as np
import time


def Binarize  (a_img,a_threshold) :

    img = np.zeros((a_img.shape[0], a_img.shape[1]),np.int)

    for i in range (a_img.shape [0]):
        for j in range (a_img.shape[1]):   
            if (a_img[i,j] > a_threshold - 1):

                img[i,j] = 255

    return img

def Downsample(img):

    new_img = np.zeros((64, 64),np.int)

    for i in range (64):
        for j in range (64):   

            new_img[i,j] = img[i*8,j*8]

    return new_img

def YokoiCalc(b, c, d, e):
    if b == c:
        if (d != b) or (e != b):
            return 'q'
        elif (d == b) and (e == b):
            return 'r'
    elif b != c:
        return 's'

def Yokoi_Single_Point(img, i, j):
    rows, columns = img.shape
    temp_img = np.zeros((rows + 2, columns + 2), np.int)
    temp_img[1:rows + 1, 1:columns + 1] = img
    new_img = np.zeros((rows, columns), np.int)
    dict_f = dict({'q': 0, 's': 0, 'r': 0})
    i += 1
    j += 1
    dict_f[YokoiCalc(temp_img[i, j], temp_img[i, j + 1], temp_img[i - 1, j + 1], temp_img[i - 1, j])] += 1
    dict_f[YokoiCalc(temp_img[i, j], temp_img[i - 1, j], temp_img[i - 1, j - 1], temp_img[i, j - 1])] += 1
    dict_f[YokoiCalc(temp_img[i, j], temp_img[i, j - 1], temp_img[i + 1, j - 1], temp_img[i + 1, j])] += 1
    dict_f[YokoiCalc(temp_img[i, j], temp_img[i + 1, j], temp_img[i + 1, j + 1], temp_img[i, j + 1])] += 1
    if dict_f['r'] is 4:
        return 5
    else:
        return dict_f['q']

def Yokoi(img):

    img_rows, img_columns = img.shape
    temp_img = np.zeros((img_rows+2, img_columns+2), np.int) 
    temp_img[1:img_rows+1, 1:img_columns+1] = img
    new_img = np.zeros((img_rows, img_columns), np.int)
    
    for i in range(1, 1+img_rows):
        for j in range(1, 1+img_columns):

            Yokoi_num = 0
            neighbor = [0,0,0,0,0,0,0,0] #下 右 上 左 右下 右上 左上 左下

            if (temp_img[i,j]==0):
                continue
            if(temp_img[i+1,j]==255): #下
                Yokoi_num +=1
                neighbor[0] = 1
            if(temp_img[i,j+1]==255): #右
                Yokoi_num +=1
                neighbor[1] = 1
            if(temp_img[i-1,j]==255): #上
                Yokoi_num +=1
                neighbor[2] = 1
            if(temp_img[i,j-1]==255): #左
                Yokoi_num +=1
                neighbor[3] = 1
            if(neighbor[0]==1 and neighbor[1]==1): #右下
                if(temp_img[i+1,j+1]==255):
                    Yokoi_num -=1
                    neighbor[4] = 1
            if(neighbor[1]==1 and neighbor[2]==1): #右上
                if(temp_img[i-1,j+1]==255):
                    Yokoi_num -=1
                    neighbor[5] = 1
            if(neighbor[2]==1 and neighbor[3]==1): #左上
                if(temp_img[i-1,j-1]==255):
                    Yokoi_num -=1
                    neighbor[6] = 1
            if(neighbor[0]==1 and neighbor[3]==1): #左下
                if(temp_img[i+1,j-1]==255):
                    Yokoi_num -=1
                    neighbor[7] = 1
            if(sum(neighbor)==8):
                Yokoi_num = 5
            new_img[i-1,j-1] = Yokoi_num

    return new_img

def Connected_Shrink(img):
    
    new_img = np.full(img.shape, False, dtype=bool)
    temp_img = Yokoi(img)
   
    rows, columns = img.shape
    for i in range(rows):
        for j in range(columns):
            if temp_img[i, j] == 1:  # or temp_img[i, j] == 0:
                new_img[i, j] = True

    return new_img


def Marked(img):
    rows, columns = img.shape
    temp_img = np.zeros((rows + 2, columns + 2), np.int)
    temp_img[1:rows + 1, 1:columns + 1] = img.copy()

    new_img = np.full(img.shape, False, dtype=bool)
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            if temp_img[i, j] == 1:
                templist = [temp_img[i][j+1], temp_img[i-1][j], temp_img[i][j-1], temp_img[i+1][j]]
                if 1 in templist:
                    new_img[i - 1, j - 1] = True
    return new_img            

    



# Write a program which counts the Yokoi connectivity number on a downsampled image(lena.bmp).

# (a) Binarize
m_lena = cv2.imread("lena.bmp",0)
m_Binarize = Binarize(m_lena,128)
print("Binarize done")
# (b) Downsample
m_Downsample = Downsample(m_Binarize)
print("Downsample done")
# (c) Yokoi connectivity number
m_processed_original_img = m_Downsample.copy()
m_final_img = m_processed_original_img.copy()

while True:
    anythingchanged = False
    yokoi = Yokoi(m_processed_original_img)
    marked_img = Marked(yokoi)
    for i in range(64):
        for j in range(64):
            if Yokoi_Single_Point(m_processed_original_img, i, j) == 1 and marked_img[i, j]:
                m_final_img[i, j] = 0
                m_processed_original_img = m_final_img.copy()
                anythingchanged = True
    if not anythingchanged:
        break
    else:
        m_processed_original_img = m_final_img.copy()


cv2.imwrite('thin.bmp', m_final_img)



