import numpy as np
import cv2
# default: 800 1000 100
length=int(input("棋盘格长，即矩阵的行：\n"))
width=int(input("棋盘格宽，即矩阵的列:\n"))


checkboard_img = np.zeros((length,width))   #这是正方形棋盘格整体大小  分辨率像素：行x列
block_size=int(input("方块像素大小（需要为长和宽的公因数）:\n"))           #方块大小：边长
black_block = np.full((block_size,block_size),255) #做一个黑色的且边长为block_size的矩阵方块
for row in range(length//block_size):        #计算行有几个黑块
    for col in range(width//block_size):     #计算列有几个黑块
        if (row+col)%2==0:
            row_begin = row*block_size
            row_end = row_begin+block_size
            col_begin = col*block_size
            col_end = col_begin+block_size
            checkboard_img[row_begin:row_end,col_begin:col_end] = black_block     #画黑块
cv2.imwrite("checker_board.jpg",checkboard_img)
row=(length//block_size)-1
col=(width//block_size)-1
print("交叉点row x col（行x列）：",row,"x",col)
# cv2.imwrite('chess.jpg',)
# cv2.imshow("checker_board",checkboard_img)
# cv2.waitKey(1)