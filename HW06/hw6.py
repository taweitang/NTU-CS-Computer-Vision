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

def show_text_image(img, scale):
    text_img = np.empty(tuple(scale*i for i in img.shape))
    rows, columns = img.shape
    text_img.fill(255)
    for i in range(0,scale*rows, scale):
        for j in range(0,scale*columns, scale):
            if img[int(i/scale),int(j/scale)] ==0:
                continue
            cv2.putText(text_img,str(img[int(i/scale),int(j/scale)]),(int(j+scale/2.2),int(i+scale/1.8)),cv2.FONT_HERSHEY_COMPLEX,2,(100,10,80),5)
    return text_img
            

            


    



# Write a program which counts the Yokoi connectivity number on a downsampled image(lena.bmp).

# (a) Binarize
m_lena = cv2.imread("lena.bmp",0)
m_Binarize = Binarize(m_lena,128)
print("Binarize done")
# (b) Downsample
m_Downsample = Downsample(m_Binarize)
print("Downsample done")
# (c) Yokoi connectivity number
s_time = time.time()
m_Yokoi = Yokoi(m_Downsample)
m_Yokoi_img = show_text_image(m_Yokoi,100)
run_time = time.time() - s_time
cv2.imwrite("Yokoi.jpg",m_Yokoi_img)
print("Yokoi done")

print("run time : " + str(run_time))

