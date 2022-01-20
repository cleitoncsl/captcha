# etapa 1
import cv2
import os
import glob
from PIL import Image

def tratar_imagens(pasta_origem, pasta_destino='ajeitado'):
    arquivos = glob.glob(f"{pasta_origem}/*")

    for arquivo in arquivos:
        imagem = cv2.imread(arquivo)

        # transformar em escala de cinza
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

        _, imagem_tratada = cv2.threshold(imagem_cinza, 127, 255, cv2.THRESH_TRUNC or cv2.THRESH_OTSU)
        nome_arquivo = os.path.basename(arquivo)
        cv2.imwrite(f'{pasta_destino}/{nome_arquivo}', imagem_tratada)

    arquivos = glob.glob(f'{pasta_destino}/*')

    for arquivo in arquivos:
        imagem = Image.open(arquivo)
        imagem = imagem.convert("L")
        imagem2 = Image.new("L", imagem.size, 255)

        for x in range(imagem.size[1]):
            for y in range(imagem.size[0]):
                cor_pixel = imagem.getpixel((y, x))
                if cor_pixel < 115:
                    imagem2.putpixel((y, x), 0)
        nome_arquivo = os.path.basename(arquivo)
        imagem2.save(f'{pasta_destino}/{nome_arquivo}')


if __name__ == "__main__":
    tratar_imagens('bdcaptcha')

