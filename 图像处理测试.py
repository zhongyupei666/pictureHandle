import numpy as np
import cv2 as cv
import scipy.io as scio
import matplotlib.pyplot as plt

# 数据图像提取的操作;
datapath = "./spect.mat"


data = scio.loadmat(datapath)
# print(data.keys())
# 最后的输出的结果为:
# dict_keys(['__header__', '__version__', '__globals__', 'spect'])
DataSpect = data['spect']
# print(DataSpect)


# 使用矩阵进行仿射变换的过程;
r,g,b,d = cv.split(DataSpect)
h,w = r.shape[:2]

# 进行仿射变换中的平移变换；
MOne = np.float32([[1,0,0],[0,1,0]])
MTwo = np.float32([[1,0,-3],[0,1,0]])
MThree = np.float32([[1,0,-6],[0,1,0]])
MFour = np.float32([[1,0,-9],[0,1,0]])


# 下面使用Cv的warpAffine(img,M,(w,h))来进行平移的操作;

rMove = cv.warpAffine(r,MOne,(w,h))
gMove = cv.warpAffine(g,MTwo,(w,h))
bMove = cv.warpAffine(b,MThree,(w,h))
dMove = cv.warpAffine(d,MFour,(w,h))

"""
rMove = np.log(np.abs(np.fft.ifft2(rMove)))
gMove = np.log(np.abs(np.fft.ifft2(gMove)))
bMove = np.log(np.abs(np.fft.ifft2(bMove)))
dMove = np.log(np.abs(np.fft.ifft2(dMove)))
"""
rMove = rMove.T
for i in range(820):
    rMove[i] = np.log(np.abs(np.fft.ifft(rMove[i])))

gMove = gMove.T
for i in range(820):
    gMove[i] = np.log(np.abs(np.fft.ifft(gMove[i])))

bMove = bMove.T
for i in range(820):
    bMove[i] = np.log(np.abs(np.fft.ifft(bMove[i])))

dMove = dMove.T
for i in range(820):
    dMove[i] = np.log(np.abs(np.fft.ifft(dMove[i])))

rMove = rMove.T
gMove = gMove.T
bMove = bMove.T
dMove = dMove.T
result = (np.abs(rMove) + np.abs(gMove) + np.abs(bMove) + np.abs(dMove))/4

print(result[1023])
plt.subplot(121)
plt.imshow(np.abs(rMove))
plt.subplot(122)
plt.imshow(result)
# plt.imshow(DataSpect)
plt.show()

