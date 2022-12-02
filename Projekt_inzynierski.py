##wczytywanie pakietow
import numpy as np
import cv2
import math
import pandas as pd
import matplotlib.pyplot as plt
##wczytywanie obrazu oraz badanie rozmiaru

image = cv2.imread("56_126_0_0_71M_0007.tif")
max_h, max_w, _ = image.shape
min_h = 0
min_w = 0
print('szerokosc', max_h)
print('wysokosc', max_w)

##maskowanie niepotrzebnych czesci obrazu
mask = np.zeros(image.shape[:2], dtype="uint8")
# lewy, górny narożnik
cv2.rectangle(mask, (min_w, min_h), (min_w + 700, min_h + 700), 255, -1)

# prawy, górny narożnik
cv2.rectangle(mask, (max_w, min_h), (max_w - 700, min_h + 700), 255, -1)

# prawy, dolny narożnik
cv2.rectangle(mask, (max_w - 800, max_h - 800), (max_w, max_h), 255, -1)

# lewy, dolny narożnik
cv2.rectangle(mask, (min_w - 800, max_h - 800), (min_w + 800, max_h), 255, -1)
image = cv2.bitwise_or(image, image, mask=mask)

##wyswietlenie obrazu z maska
# cv2.imshow("Zamaskowany obraz",image)


# konwersja do obrazu w skali szarosci
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('g', gray)

# detekcja krawedzi
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# wykrywanie linii

lines = cv2.HoughLinesP(
    edges,
    1,
    np.pi / 90,  # w radianach
    threshold=100,
    minLineLength=160,
    maxLineGap=10
)

px1 = []
py1 = []
px2 = []
py2 = []
for points in lines:
    x1, y1, x2, y2 = points[0]
    # rysowanie linii
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    px1.append(x1)
    py1.append(y1)
    px2.append(x2)
    py2.append(y2)


# funkcja obliczania kąta między prostymi
def dot(vA, vB):
    return vA[0] * vB[0] + vA[1] * vB[1]


def ang(lineA, lineB):

    vA = [(lineA[0][0] - lineA[1][0]), (lineA[0][1] - lineA[1][1])]
    vB = [(lineB[0][0] - lineB[1][0]), (lineB[0][1] - lineB[1][1])]

    dot_prod = dot(vA, vB)

    magA = dot(vA, vA) ** 0.5
    magB = dot(vB, vB) ** 0.5

    cos_ = dot_prod / magA / magB
    angle = math.acos(dot_prod / magB / magA)
    ang_deg = math.degrees(angle) % 360

    if ang_deg - 180 >= 0:

        return 360 - ang_deg
    else:

        return ang_deg


# fukncja przeciecia
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        q = ((a[0] * b[1]) - (a[1] * b[0]))
        return q

    div = det(xdiff, ydiff)
    if div == 0:
        return False

    else:
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y
t = len(px1)
lista = []
for j in range(0,t):
    E = [px1[j], py1[j]]
    F = [px2[j], py2[j]]
    if (j < t-1):
        G = [px1[j+1], py1[j+1]]
        H = [px2[j+1], py2[j+1]]
        if (ang([E, F], [G, H]) <= 95) and (ang([E, F], [G, H]) >= 85):

            A, B, C, D = E, F, G, H
            lista.append(line_intersection([A, B], [C, D]))
    else:
        break
lista1 = []
print(lista)


df_a = pd.DataFrame(lista)
df_a.columns = 'x', 'y'
print(df_a)

plt.xlabel('x')
plt.ylabel('y')
plt.scatter(df_a.x, df_a.y, c="green")


plt.show()




