# -*- coding: utf-8 -*-
"""
Created on Mon May 23 12:49:28 2022

@author: Samara
"""
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# le uma imagem
imagem_lida = cv.imread('imagem_Ma_Quiteria1.jpg')
print(type(imagem_lida))
plt.imshow(imagem_lida), plt.show()

# converte para preto e branco e gera linha dos pixels
gray = cv.cvtColor(imagem_lida, cv.COLOR_BGR2GRAY)
print(type(gray), gray.shape)
ravel = gray.ravel()
print(type(ravel), ravel.shape)
linha_imagem_lida = ravel.astype(int)
print(type(linha_imagem_lida), linha_imagem_lida.shape)

# le outra imagem
imagem_lida2 = cv.imread('imagem_Ma_Quiteria2.jpg')
print(type(imagem_lida2))
plt.imshow(imagem_lida2), plt.show()

# converte para preto e branco e gera linha dos pixels
gray2 = cv.cvtColor(imagem_lida2, cv.COLOR_BGR2GRAY)
print(type(gray2), gray2.shape)
ravel2 = gray2.ravel()
print(type(ravel2), ravel2.shape)
linha_imagem_lida2=ravel2.astype(int)
print(type(linha_imagem_lida2), linha_imagem_lida2.shape)

# junta as linhas dos pixels de uma imagem e de outra
matriz = np.stack((linha_imagem_lida, linha_imagem_lida2), axis=1)
print('Matriz:')
print(type(matriz), matriz.shape)

imagens_em_conjunto = pd.DataFrame(matriz)
print('Imagens em conjunto:')
print(type(imagens_em_conjunto), imagens_em_conjunto.shape)

# calcula a correlaçao dos pixels das duas imagens
correlacao = imagens_em_conjunto.corr(method='pearson')
print('Correlação:')
print(type(correlacao), correlacao.shape)
print(correlacao)

# plota a correlaçao dos pixels das duas imagens
plt.figure(figsize=(70,70))
sns.heatmap(correlacao)
plt.show()

# calcula a covariancia, acho que posso retirar essa parte
covariancia = np.cov(linha_imagem_lida, linha_imagem_lida2)
print('Covariância:')
print(type(covariancia), covariancia.shape)
print(covariancia)

# exibe o pedaço de a até b dos pixelsets das duas imagens simultaneamente:
a = 50630
b= 50730
print(matriz[a:b,:])




