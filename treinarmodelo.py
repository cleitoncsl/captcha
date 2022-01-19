import cv2
import os
import numpy as np
import pickle
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from helpers import resize_to_fit

dados = []
rotulos = []
pasta_base_imagens = "base_letras"

imagens = paths.list_images(pasta_base_imagens)
# print(list(imagens)) isso aqui é legal. lista os arquivos

for arquivo in imagens:
    rotulo = arquivo.split(os.path.sep)[-2]
    imagem = cv2.imread(arquivo)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # padronizar imagem em 20x20

    imagem = resize_to_fit(imagem, 20, 20)

    # adicionar uma dimensão para o keras ler a imagem
    imagem = np.expand_dims(imagem, axis=2)

    # adicionar listas de dados e rotulos
    rotulos.append(rotulo)
    dados.append(imagem)

dados = np.array(dados, dtype="float") / 255
rotulos = np.array(rotulos)

# separação em dados de treino %75 e dados de teste %25
(X_train, X_test, Y_train, Y_test) = train_test_split(dados, rotulos, test_size=0.25, random_state=0)

# Converter para one-hot encoding
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# salvar o labelbinaryzer em um arquivo pickle
with open('rotulos_modelo.dat', 'wb') as arquivo_pickle:
    pickle.dump(lb, arquivo_pickle)

# criar e treinar a inteligencia artificial
modelo = Sequential()

# criar Primeira as camadas da rede neural
modelo.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# criar segunda camada de rede neural
modelo.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# criar terceira camada
modelo.add(Flatten())
modelo.add(Dense(500, activation="relu"))

# criar camada de saida
modelo.add(Dense(26, activation="softmax"))

# criar todas as camadas
modelo.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# treinar inteligencia artificial
modelo.fit(X_train, Y_train, validation_data=(X_test, Y_test),batch_size=26, epochs=10, verbose=1)

# salvar o modelo em um arquivo

modelo.save("modelo_treinado.hdf5")
