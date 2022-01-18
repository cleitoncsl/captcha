import cv2
from PIL import Image

metodos = [
    cv2.THRESH_BINARY,
    cv2.THRESH_BINARY_INV,
    cv2.THRESH_TRUNC,
    cv2.THRESH_TOZERO,
    cv2.THRESH_TOZERO_INV
]

imagem = cv2.imread("bdcaptcha/telanova0.png")

# transformar em escalda de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

i = 0
for metodo in metodos:
    i += 1
    _, imagem_tratada = cv2.threshold(imagem_cinza, 127, 255, metodo or cv2.THRESH_OTSU)
    cv2.imwrite(f'testesmetodo/imagem_tratada_{i}.png', imagem_tratada)

imagem = Image.open("testesmetodo/imagem_tratada_3.png")


