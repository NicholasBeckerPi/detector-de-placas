import numpy as np
import cv2  as cv
import sys


def ImgWorkon():
    global img, canny_o
    img = cv.imread('img/placa_pare_3.jpeg')

    # Redeimensiona a imagem
    max_dimension = max(img.shape)
    scale = 400/max_dimension
    img = cv.resize(img, None, fx=scale, fy=scale)

    # Aplica filtros
    img_blur = cv.GaussianBlur(img,(5, 5), 0)
    img_gray = cv.cvtColor(img_blur, cv.COLOR_BGR2GRAY)
    a, thresh = cv.threshold(img_gray, 127, 255, 0)

    # Detecção de bordas. 1º arg é a img input, 2º é o valMin de gradiente (abaixo disso não é borda),
    # 3º é maxVal de gradiente (acima disso é borda)
    canny_o = cv.Canny(thresh, 100, 110)

    # Acha os contornos e desenha-os. OBS.: RETR_EXTERNAL procura contornos externos, ignorando os
    # que tem hierarquia mais alta
    countours, hierarchy = cv.findContours(canny_o, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print(hierarchy)
    #cv.drawContours(img ,countours, 0, (255, 0,0), 7)

    # Iterar sobre o countours
    for cnt in countours:
        # Calcula o perimetro da figura
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.04 * peri, True)
        x, y, w, h = cv.boundingRect(approx)
        rectangle_img = cv.rectangle(img, (x, y), (x + w , y + h), (0, 255, 0), 8)
        cv.putText(rectangle_img, f'PONTOS:{len(approx)}', (x, y-10),
        cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def Window():
    cv.imshow('CONTORNOS',img)
    cv.imshow('LINHAS E NEGATIVO',canny_o)
    save_key = cv.waitKey(0)
    if save_key == ord('s'):
        cv.imwrite('./img/altered_img2.png', img)

if __name__ == '__main__':
    ImgWorkon()
    Window()
