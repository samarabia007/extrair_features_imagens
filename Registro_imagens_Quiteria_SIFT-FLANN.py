# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 18:09:41 2022

@author: Samara
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ler duas imagens
img1 = cv.imread('Quiteria_RX_media_resolucao.jpg')
img2 = cv.imread('Quiteria_IR_media_resolucao.jpg')

# print das duas imagens e seus tipos/formatos
plt.imshow(img1),plt.show() 
print('Imagem 1:', type(img1))
print(img1.shape)

plt.imshow(img2),plt.show()
print('Imagem 2:', type(img2))
print(img2.shape)

# converte para preto e branco
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
print(type(gray1))
print(gray1.shape)

gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
print(type(gray2))
print(gray2.shape)

# dar o SIFT nelas

#keypoints img1
sift = cv.xfeatures2d.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(gray1,None)

# desenhar os keypoints img1
img3 = cv.drawKeypoints(gray1,keypoints_1,img1)
print(type(img3))
print(img3.shape)
plt.imshow(img3),plt.show()

#keypoints img1
#sift2 = cv.xfeatures2d.SIFT_create()
keypoints_2, descriptors_2 = sift.detectAndCompute(gray2,None)

# desenhar os keypoints img2
img4 = cv.drawKeypoints(gray2,keypoints_2,img2)
print(type(img4))
print(img4.shape)
plt.imshow(img4),plt.show()

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors_1,descriptors_2,k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

MIN_MATCH_COUNT = 10

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ keypoints_1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints_2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = gray1.shape 
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    gray2 = cv.polylines(gray2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    transformed_img = cv.warpPerspective(gray1, M, (539,872))
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)


# salva moldura datransformacao da imagem
cv.imwrite('transformacao_por_SIFT.jpg', gray2)
plt.imshow(gray2), plt.show()

# salva imagem transformada
cv.imwrite('imagem_transformada_por_SIFT.jpg', transformed_img)
plt.imshow(transformed_img), plt.show()

# cria imagem dos macthes e a salva
matches_image = cv.drawMatches(img1,keypoints_1,img2,keypoints_2,good,None,**draw_params)
plt.imshow(matches_image, 'gray'),plt.show()

matches_image = Image.fromarray(matches_image)

matches_image.save('matches.png', format='PNG')

matches_image.show()