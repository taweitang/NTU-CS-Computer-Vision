import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageDraw
from PIL import Image

def Binarize  (a_img,a_threshold) :

    img = np.zeros((a_img.shape[0], a_img.shape[1],1),dtype=np.uint8)

    for i in range (a_img.shape [0]):
        for j in range (a_img.shape[1]):   
            if (a_img[i,j,0] > a_threshold - 1):

                img[i,j,0] = 255

    return img

def Histogram (a_img) :

    histogram = np.zeros(256)

    for i in range (a_img.shape [0]):
        for j in range (a_img.shape[1]):
            
            gray = int(a_img[i,j,0])
            histogram[gray] += 1

    return histogram

def Connected (a_img) :

    labels = np.zeros((a_img.shape[0], a_img.shape[1]),np.int)
    label = 1
    # First pass
    for i in range (a_img.shape [0]):
        for j in range (a_img.shape[1]):
            if (a_img[i,j,0] > 0):
                labels[i,j] = label
                label += 1
    
    # Second pass
    repeat = True
    loop = 1
    while (repeat == True):
        print("loop : " + str(loop))
        loop+=1
        repeat = False
        # top down
        for i in range (a_img.shape [0]):
            for j in range (a_img.shape[1]):
                if (labels[i,j] > 0):
                    if(i>0 and labels[i,j]*labels[i-1,j]>0):
                        if (labels[i,j]!=labels[i-1,j]):
                            repeat = True
                        m = min(labels[i,j],labels[i-1,j])
                        labels[i,j] = m
                        labels[i-1,j] = m

                    if(i<511 and labels[i,j]*labels[i+1,j]>0):
                        if (labels[i,j]!=labels[i+1,j]):
                            repeat = True
                        m = min(labels[i,j],labels[i+1,j])
                        labels[i,j] = m
                        labels[i+1,j] = m
                        
                    if(j>0 and labels[i,j]*labels[i,j-1]>0):
                        if (labels[i,j]!=labels[i,j-1]):
                            repeat = True
                        m = min(labels[i,j],labels[i,j-1])
                        labels[i,j] = m
                        labels[i,j-1] = m
                        
                    if(j<511 and labels[i,j]*labels[i,j+1]>0):
                        if (labels[i,j]!=labels[i,j+1]):
                            repeat = True
                        m = min(labels[i,j],labels[i,j+1])
                        labels[i,j] = m
                        labels[i,j+1] = m
                        

         # bottom up
        for i in range (a_img.shape [0]):
            for j in range (a_img.shape[1]):
                if (labels[511-i,511-j] > 0):
                    if(i<511 and labels[511-i,511-j]*labels[511-i-1,511-j]>0):
                        if (labels[511-i,511-j]!=labels[511-i-1,511-j]):
                            repeat = True
                        m = min(labels[511-i,511-j],labels[511-i-1,511-j])
                        labels[511-i,511-j] = m
                        labels[511-i-1,511-j] = m
                        
                    if(i>0 and labels[511-i,511-j]*labels[511-i+1,511-j]>0):
                        if (labels[511-i,511-j]!=labels[511-i+1,511-j]):
                            repeat = True
                        m = min(labels[511-i,511-j],labels[511-i+1,511-j])
                        labels[511-i,511-j] = m
                        labels[511-i+1,511-j] = m
                        
                    if(j<511 and labels[511-i,511-j]*labels[511-i,511-j-1]>0):
                        if (labels[511-i,511-j]!=labels[511-i,511-j-1]):
                            repeat = True
                        m = min(labels[511-i,511-j],labels[511-i,511-j-1])
                        labels[511-i,511-j] = m
                        labels[511-i,511-j-1] = m
                        
                    if(j>0 and labels[i,j]*labels[511-i,511-j+1]>0):
                        if (labels[511-i,511-j]!=labels[511-i,511-j+1]):
                            repeat = True
                        m = min(labels[511-i,511-j],labels[511-i,511-j+1])
                        labels[511-i,511-j] = m
                        labels[511-i,511-j+1] = m
                        

    print("Second pass done")   

    #計算有哪些是>500個pixels的群體
    pixelcount = np.zeros(133960, np.int)
    area = np.zeros(0, np.int)

    for i in range (512):
        for j in range (512):
            if labels[i,j] != 0:
                pixelcount[labels[i,j]] += 1

    #找出pixel數大於500之區塊            
    for i in range (133960):
        if pixelcount[i] > 500:
            area = np.append(area, i)

    #繪製矩形
    im_binary = Image.new("L", (512,512), 0) 
    for i in range (512):
        for j in range (512):
            value = int(a_img[i,j]) 
            im_binary.putpixel((j,i), value)
    boundary = np.zeros([np.size(area), 4],np.int)
    connected_component = im_binary.convert('RGB')
    draw = ImageDraw.Draw(connected_component)
    for k in range(np.size(area)):
        minx, miny, maxx, maxy = 600, 600, 0, 0
        A,xbar,ybar = 0,0,0
        for i in range (512):
            for j in range (512):
                if labels[i,j] == area[k]:
                    A += 1
                    xbar += j
                    ybar += i

                    if j < minx:
                        minx = j
                    elif j > maxx:
                        maxx = j
                    if i < miny:
                        miny = i
                    elif i > maxy:
                        maxy = i
        xbar = int(xbar/A)
        ybar = int(ybar/A)
        draw.rectangle([minx, miny, maxx, maxy],fill=None, outline="red")
        draw.line(((xbar-7,ybar),(xbar+7,ybar)),'red',5)
        draw.line(((xbar,ybar-7),(xbar,ybar+7)),'red',5)
    print("Draw done")
    #儲存connected_component之圖片
    connected_component.save("connected_component.bmp")
    print("Save done")


# load and process
m_lena = cv2.imread("lena.bmp")
m_Binarize = Binarize(m_lena,128)
Connected(m_Binarize)
m_Histogram = Histogram(m_lena)
plt.bar(range(1,257), m_Histogram)

# save
plt.savefig("Histogram.jpg")
cv2.imwrite('Binarize.bmp', m_Binarize)

# show
cv2.imshow("lena",m_lena)
cv2.imshow("Binarize",m_Binarize)
cv2.waitKey(0)
cv2.destroyAllWindows()