# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 22:45:29 2022

@author: Samara
"""
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ler duas imagens
img1 = cv.imread('recorte1.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('recorte2.jpg', cv.IMREAD_GRAYSCALE)

# dar o ORB nelas
orb = cv.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# dar o Brute Force Matcher e os descriptors bf.Match(des1, des2)
bf = cv.BFMatcher (cv.NORM_HAMMING, crossCheck = True)
matches = bf.match(des1, des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# cria variáveis com uma das dimensões do tamanho do número de matches 
matches = matches[:int(len(matches)*90)]
no_of_matches = len(matches)
p1 = np.zeros((no_of_matches, 2))
p2 = np.zeros((no_of_matches, 2))

# Atribui o queryIdx dos kp1 dos matches da img1 à variável p1 e os trainIdx da img2 (à variável p2)
for i in range(len(matches)):
    p1[i, :] = kp1[matches[i].queryIdx].pt
    p2[i, :] = kp2[matches[i].trainIdx].pt
    
#cria máscara, homografia e imagem transformada por elas
homography, mask = cv.findHomography(p1, p2, cv.RANSAC)
transformed_img = cv.warpPerspective(img2, homography, (919,1119))
cv.imwrite('imagem_transformada.jpg', transformed_img)
                                     

# Draw first 10 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
img3 = Image.fromarray(img3)
img3.save('img3.png', format = 'PNG')
# [print da img3]


