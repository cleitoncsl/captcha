import cv2
import os
import glob
arquivos = glob.glob('ajeitado/*')

for arquivo in arquivos:
    imagem = cv2.imread(arquivo)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

    # em preto e branco
    _, imagem = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY_INV)

    # Encontrar os contornos
    contornos, _ = cv2.findContours(imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regiao_letras = []

    #Filtro Contornos
    for contorno in contornos:
        (x, y, largura, altura) = cv2.boundingRect(contorno)
        area = cv2.contourArea(contorno)
        if area > 115:
            regiao_letras.append((x, y , largura, altura))

    if len(regiao_letras) != 5:
        continue

    # Desenhar os contornos e separar as letras em arquivos individuais

    imagem_final = cv2.merge(imagem * 3)

    i = 0
    for retangulo in regiao_letras:
        x, y, largura, altura = retangulo
        imagem_letra = imagem[y-2:y+altura+2, x-2:x+largura+2]
        i += 1
        nome_arquivo = os.path.basename(arquivo).replace(".png", f"_letra{i}.png")
        cv2.imwrite(f'letras/{nome_arquivo}', imagem_letra)
        cv2.rectangle(imagem_final, (x - 2, y - 2), (x + largura + 2, y + altura + 2), (0,255,0), 1)
    nome_arquivo = os.path.basename(arquivo)
    cv2.imwrite(f"identificados/{nome_arquivo}", imagem_final)
