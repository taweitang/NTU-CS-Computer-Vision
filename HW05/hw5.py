import cv2
import numpy as np


m_kernel = np.array([[0,1,1,1,0],
                    [1,1,1,1,1],
                    [1,1,1,1,1],
                    [1,1,1,1,1],
                    [0,1,1,1,0]])

def Dilation(img, ker):

    img_rows, img_columns = img.shape
    ker_rows, ker_columns = ker.shape
    row_dist, column_dist = int((ker_rows-1)/2), int((ker_columns-1)/2)
    temp_img = np.zeros((img_rows+2*row_dist, img_columns+2*column_dist), np.int) 
    temp_img[row_dist:img_rows+row_dist, column_dist:img_columns+column_dist] = img
    new_img = np.zeros((img_rows+2*row_dist, img_columns+2*column_dist), np.int)   
    kernel_flip = np.flip(ker,0)
    
    for i in range(row_dist, img_rows+row_dist):
        for j in range(column_dist, img_columns+column_dist):
            new_img[i, j] = np.nanmax(temp_img[i-row_dist: i+row_dist+1, j-column_dist: j+column_dist+1]+kernel_flip)
    new_img = new_img[row_dist:img_rows+row_dist, column_dist:img_columns+column_dist]      
    
    return new_img
    
def Erosion(img, ker):
    img_rows, img_columns = img.shape
    ker_rows, ker_columns = ker.shape
    row_dist, column_dist = int((ker_rows-1)/2), int((ker_columns-1)/2)
    temp_img = 255 * np.ones((img_rows+2*row_dist, img_columns+2*column_dist), np.int) 
    temp_img[row_dist:img_rows+row_dist, column_dist:img_columns+column_dist] = img
    new_img = 255*np.ones((img_rows+2*row_dist, img_columns+2*column_dist), np.int)   
    
    for i in range(row_dist, img_rows+row_dist):
        for j in range(column_dist, img_columns+column_dist):
            new_img[i, j] = np.nanmin(temp_img[i-row_dist: i+row_dist+1, j-column_dist: j+column_dist+1]-ker)

    new_img = new_img[row_dist:img_rows+row_dist, column_dist:img_columns+column_dist]      
    return new_img
    
def Opening(img, ker):
    new_img = Dilation(Erosion(img, ker), ker)
    return new_img

def Closing(img, ker):
    new_img = Erosion(Dilation(img, ker), ker)
    return new_img



# Write a program to generate images and histograms:

# (a) Dilation
m_lena = cv2.imread("lena.bmp",0)
m_Dilation = Dilation(m_lena,m_kernel)
cv2.imwrite("Dilation.jpg",m_Dilation)
print("Dilation done")
# (b) Erosion
m_Erosion = Erosion(m_lena,m_kernel)
cv2.imwrite("Erosion.jpg",m_Erosion)
print("Erosion done")
# (c) Opening
m_Opening = Opening(m_lena,m_kernel)
cv2.imwrite("Opening.jpg",m_Opening)
print("Opening done")
# (d) Closing
m_Closing = Closing(m_lena,m_kernel)
cv2.imwrite("Closing.jpg",m_Closing)
print("Closing done")


