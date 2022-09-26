# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 22:24:41 2022

@author: Samara
"""


import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image


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


# fazer um "for" das diferentes regiões da imagens_em_conjunto
a=[]
b=[]
valores = [0 , 1000 , 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
vetor = []
classe = []
for a in valores:
    b = a + 999
    pedaco = matriz[a:b,:]
    print(type(pedaco), (pedaco.shape))

# calcula a correlaçao dos pixels das duas imagens
    dfpedaco = pd.DataFrame(pedaco)
    correlacao = dfpedaco.corr(method='pearson')
    print('Correlação:')
    print(type(correlacao), correlacao.shape)
    componente_vetor = correlacao[1]
    print(componente_vetor[0])
    vetor.append(componente_vetor[0])
    if componente_vetor[0] > 0.4 : 
        classe_da_regiao = 1
    else:
        classe_da_regiao = 0
    classe.append(classe_da_regiao)
    
    # tornar as regioes classe 1 pretas numa nova matriz (matriz_filtrada)
    todos_os_pedacos = []

    if classe_da_regiao == 1:
        pedaco = 0
        todos_os_pedacos.append(pedaco)
    
    else:
        todos_os_pedacos.append(pedaco)

# printar as correlações e as classes de todas as regiões em listas
print(vetor)
print(classe)

#matriz_filtrada = todos_os_pedacos.values
print (type(todos_os_pedacos))

# separar as duas colunas da matriz filtrada para recuperar cada uma das imagens
# matriz_duma_imagem_filtrada = # dar um desstack (matriz_filtrada) e pegar uma coluna só
# matriz_doutra_imagem_filtrada = # dar um desstack (matriz_filtrada) e pegar a outra coluna

# redimensionar as duas colunas da matriz, separadas novamente, para o tamanho da imagem original
# pesquisar como fazer isso!!

#imagem_filtrada1 = Image.fromarray(matriz_duma_imagem_filtrada))
#print(type(imagem_filtrada1), imagem_filtrada1.shape)
#plt.imshow(imagem_filtrada1), plt.show()

#imagem_filtrada2 = Image.fromarray(matriz_doutra_imagem_filtrada))
#print(type(imagem_filtrada2), imagem_filtrada2.shape)
#plt.imshow(imagem_filtrada2), plt.show()

