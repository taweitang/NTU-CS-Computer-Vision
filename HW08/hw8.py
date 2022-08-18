import cv2
import numpy as np

original_img = cv2.imread('lena.bmp', 0)

def gaussianNoise(img, amp):

    noise = np.random.normal(loc = 0, scale = 1, size = img.shape)
    noisy_img = img + amp*noise
    return  noisy_img

def saltpepperNoise(img, prob):

    noisy_img = img.copy()
    noise = np.random.uniform(low=0, high=1, size = img.shape)

    for i, j in np.ndindex(noise.shape):
        if noise[i,j]<prob:
            noisy_img[i,j] = 0
        elif noise[i,j]> 1-prob:
            noisy_img[i, j] = 255
        else:
            pass
    return  noisy_img

def boxFiltering(img, boxsize):

    img_rows, img_columns = img.shape
    ker_rows = ker_columns = boxsize
    box = np.full((boxsize, boxsize), 1, dtype=int)

    row_dist, column_dist = int((ker_rows - 1) / 2), int((ker_columns - 1) / 2)

    temp_img = np.full((img_rows + 2 * row_dist, img_columns + 2 * column_dist), -1)

    temp_img[row_dist:img_rows + row_dist, column_dist:img_columns + column_dist] = img.copy()

    new_img = np.zeros((img_rows, img_columns), np.int)

    for i in range(row_dist, img_rows + row_dist):
        for j in range(column_dist, img_columns + column_dist):

            temp = temp_img[i - row_dist: i + row_dist + 1, j - column_dist: j + column_dist + 1]

            unique, counts = np.unique(temp, return_counts=True)
            dict4den = dict(zip(unique, counts))
            if -1 in dict4den:
                temp2 = temp.copy()
                for i2 in range(boxsize):
                    for j2 in range(boxsize):
                        if temp[i2, j2] == -1:
                            temp2[i2, j2] = 0
                num = np.sum(np.multiply(box, temp2))
                den = boxsize ** 2 - dict4den[-1]
            else:
                num = np.sum(np.multiply(box, temp))
                den = boxsize ** 2
            new_img[i - row_dist, j - column_dist] = num / den
    return new_img


def medianFiltering(img, boxsize):

    img_rows, img_columns = img.shape
    ker_rows = ker_columns = boxsize
    row_dist, column_dist = int((ker_rows - 1) / 2), int((ker_columns - 1) / 2)
    temp_img = np.full((img_rows + 2 * row_dist, img_columns + 2 * column_dist), -1)
    temp_img[row_dist:img_rows + row_dist, column_dist:img_columns + column_dist] = img.copy()
    new_img = np.zeros((img_rows, img_columns), np.int)
    for i in range(row_dist, img_rows + row_dist):
        for j in range(column_dist, img_columns + column_dist):
            temp = temp_img[i - row_dist: i + row_dist + 1, j - column_dist: j + column_dist + 1]
            unique, counts = np.unique(temp, return_counts=True)
            dict4den = dict(zip(unique, counts))
            if -1 in dict4den:
                temp2 = np.array([])
                for i2 in range(boxsize):
                    for j2 in range(boxsize):
                        if temp[i2, j2] == -1:
                            pass
                        else:
                            temp2 = np.append(temp2, temp[i2, j2])
                m = np.sort(temp2, axis=None)
                new_img[i - row_dist, j - column_dist] = m[ int((m.size - 1) / 2)]
            else:
                m = np.sort(temp, axis=None)
                new_img[i - row_dist, j - column_dist] = m[ int((m.size-1) / 2)]
    return new_img


def GrayScale_Dilation(img, ker):
    img_rows, img_columns = img.shape
    ker_rows, ker_columns = ker.shape
    row_dist, column_dist = int((ker_rows - 1) / 2), int((ker_columns - 1) / 2)
    temp_img = np.zeros((img_rows + 2 * row_dist, img_columns + 2 * column_dist), np.int)
    temp_img[row_dist:img_rows + row_dist, column_dist:img_columns + column_dist] = img
    new_img = np.zeros((img_rows + 2 * row_dist, img_columns + 2 * column_dist), np.int)
    kernel_flip = np.flip(ker,axis = 0)

    for i in range(row_dist, img_rows + row_dist):
        for j in range(column_dist, img_columns + column_dist):
            new_img[i, j] = np.nanmax(
                temp_img[i - row_dist: i + row_dist + 1, j - column_dist: j + column_dist + 1] + kernel_flip)
    new_img = new_img[row_dist:img_rows + row_dist, column_dist:img_columns + column_dist]

    return new_img


def GrayScale_Erosion(img, ker):
    img_rows, img_columns = img.shape
    ker_rows, ker_columns = ker.shape
    row_dist, column_dist = int((ker_rows - 1) / 2), int((ker_columns - 1) / 2)
    temp_img = 255 * np.ones((img_rows + 2 * row_dist, img_columns + 2 * column_dist), np.int)
    temp_img[row_dist:img_rows + row_dist, column_dist:img_columns + column_dist] = img
    new_img = 255 * np.ones((img_rows + 2 * row_dist, img_columns + 2 * column_dist), np.int)
    for i in range(row_dist, img_rows + row_dist):
        for j in range(column_dist, img_columns + column_dist):
            new_img[i, j] = np.nanmin(
                temp_img[i - row_dist: i + row_dist + 1, j - column_dist: j + column_dist + 1] - ker)

    new_img = new_img[row_dist:img_rows + row_dist, column_dist:img_columns + column_dist]
    return new_img


def GrayScale_Opening(img, ker):
    return GrayScale_Dilation(GrayScale_Erosion(img, ker), ker)

def GrayScale_Closing(img, ker):
    return GrayScale_Erosion(GrayScale_Dilation(img, ker), ker)

def op_cl_Filtering(img, ker):
    return GrayScale_Closing(GrayScale_Opening(img, ker), ker)

def cl_op_Filtering(img, ker):
    return GrayScale_Opening(GrayScale_Closing(img, ker), ker)


def vsCalc(img):
    rows, cols = img.shape
    sum1=sum2=0
    img = img/255

    for i in range(rows):
        for j in range(cols):
            sum1 += img[i, j]
    mu = sum1 / (rows*cols)

    for i in range(rows):
        for j in range(cols):
            sum2 += ((img[i, j]-mu) ** 2)
    vs = sum2 / (rows*cols)
    return vs

def vnCalc(o_img, n_img):
    o_img = o_img/255
    n_img = n_img/255
    sum1 = sum2 = 0
    rows, cols = o_img.shape
    for i in range(rows):
        for j in range(cols):
            sum1 += (n_img[i, j]-o_img[i, j])
    mun = sum1 / (rows*cols)
    for i in range(rows):
        for j in range(cols):
            sum2 += ((n_img[i, j]-o_img[i, j]-mun) ** 2)
    vn = sum2 / (rows*cols)
    return vn

VS_orginal_img  = vsCalc(original_img)
kernel = np.array([[np.nan, 0, 0, 0, np.nan], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [np.nan, 0, 0, 0, np.nan]])


# noise
gaussian10 = gaussianNoise(original_img, 10)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, gaussian10))))
cv2.imwrite('gaussian10_snr'+SNR+'.bmp', gaussian10)
gaussian30 = gaussianNoise(original_img, 30)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, gaussian30))))
cv2.imwrite('gaussian30_snr'+SNR+'.bmp', gaussian30)
saltpepper005 = saltpepperNoise(original_img, 0.05)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, saltpepper005))))
cv2.imwrite('saltpepper005_snr'+SNR+'.bmp', saltpepper005)
saltpepper01 = saltpepperNoise(original_img, 0.1)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, saltpepper01))))
cv2.imwrite('saltpepper01_snr'+SNR+'.bmp', saltpepper01)

# gaussian10
gaussian10_33box = boxFiltering(gaussian10, 3)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, gaussian10_33box))))
cv2.imwrite('gaussian10_33box_snr'+SNR+'.bmp', gaussian10_33box)
gaussian10_55box = boxFiltering(gaussian10, 5)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, gaussian10_55box))))
cv2.imwrite('gaussian10_55box_snr'+SNR+'.bmp', gaussian10_55box)
gaussian10_33median = medianFiltering(gaussian10, 3)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, gaussian10_33median))))
cv2.imwrite('gaussian10_33median_snr'+SNR+'.bmp', gaussian10_33median)
gaussian10_55median = medianFiltering(gaussian10, 5)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, gaussian10_55median))))
cv2.imwrite('gaussian10_55median_snr'+SNR+'.bmp', gaussian10_55median)
gaussian10_op_cl = op_cl_Filtering(gaussian10, kernel)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, gaussian10_op_cl))))
cv2.imwrite('gaussian10_op_cl_snr'+SNR+'.bmp', gaussian10_op_cl)
gaussian10_cl_op = cl_op_Filtering(gaussian10, kernel)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, gaussian10_cl_op))))
cv2.imwrite('gaussian10_cl_op_snr'+SNR+'.bmp', gaussian10_cl_op)

# gaussian30
gaussian30_33box = boxFiltering(gaussian30, 3)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, gaussian30_33box))))
cv2.imwrite('gaussian30_33box_snr'+SNR+'.bmp', gaussian30_33box)
gaussian30_55box = boxFiltering(gaussian30, 5)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, gaussian30_55box))))
cv2.imwrite('gaussian30_55box_snr'+SNR+'.bmp', gaussian30_55box)
gaussian30_33median = medianFiltering(gaussian30, 3)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, gaussian30_33median))))
cv2.imwrite('gaussian30_33median_snr'+SNR+'.bmp', gaussian30_33median)
gaussian30_55median = medianFiltering(gaussian30, 5)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, gaussian30_55median))))
cv2.imwrite('gaussian30_55median_snr'+SNR+'.bmp', gaussian30_55median)
gaussian30_op_cl = op_cl_Filtering(gaussian30, kernel)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, gaussian30_op_cl))))
cv2.imwrite('gaussian30_op_cl_snr'+SNR+'.bmp', gaussian30_op_cl)
gaussian30_cl_op = cl_op_Filtering(gaussian30, kernel)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, gaussian30_cl_op))))
cv2.imwrite('gaussian30_cl_op_snr'+SNR+'.bmp', gaussian30_cl_op)

# saltpepper005
saltpepper005_33box = boxFiltering(saltpepper005, 3)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, saltpepper005_33box))))
cv2.imwrite('saltpepper005_33box_snr'+SNR+'.bmp', saltpepper005_33box)
saltpepper005_55box = boxFiltering(saltpepper005, 5)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, saltpepper005_55box))))
cv2.imwrite('saltpepper005_55box_snr'+SNR+'.bmp', saltpepper005_55box)
saltpepper005_33median = medianFiltering(saltpepper005, 3)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, saltpepper005_33median))))
cv2.imwrite('saltpepper005_33median_snr'+SNR+'.bmp', saltpepper005_33median)
saltpepper005_55median = medianFiltering(saltpepper005, 5)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, saltpepper005_55median))))
cv2.imwrite('saltpepper005_55median_snr'+SNR+'.bmp', saltpepper005_55median)
saltpepper005_op_cl = op_cl_Filtering(saltpepper005, kernel)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, saltpepper005_op_cl))))
cv2.imwrite('saltpepper005_op_cl_snr'+SNR+'.bmp', saltpepper005_op_cl)
saltpepper005_cl_op = cl_op_Filtering(saltpepper005, kernel)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, saltpepper005_cl_op))))
cv2.imwrite('saltpepper005_cl_op_snr'+SNR+'.bmp', saltpepper005_cl_op)

# saltpepper01
saltpepper01_33box = boxFiltering(saltpepper01, 3)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, saltpepper01_33box))))
cv2.imwrite('saltpepper01_33box_snr'+SNR+'.bmp', saltpepper01_33box)
saltpepper01_55box = boxFiltering(saltpepper01, 5)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, saltpepper01_55box))))
cv2.imwrite('saltpepper01_55box_snr'+SNR+'.bmp', saltpepper01_55box)
saltpepper01_33median = medianFiltering(saltpepper01, 3)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, saltpepper01_33median))))
cv2.imwrite('saltpepper01_33median_snr'+SNR+'.bmp', saltpepper01_33median)
saltpepper01_55median = medianFiltering(saltpepper01, 5)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, saltpepper01_55median))))
cv2.imwrite('saltpepper01_55median_snr'+SNR+'.bmp', saltpepper01_55median)
saltpepper01_op_cl = op_cl_Filtering(saltpepper01, kernel)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, saltpepper01_op_cl))))
cv2.imwrite('saltpepper01_op_cl_snr'+SNR+'.bmp', saltpepper01_op_cl)
saltpepper01_cl_op = cl_op_Filtering(saltpepper01, kernel)
SNR = str(20 * np.log10(np.sqrt(VS_orginal_img / vnCalc(original_img, saltpepper01_cl_op))))
cv2.imwrite('saltpepper01_cl_op_snr'+SNR+'.bmp', saltpepper01_cl_op)

