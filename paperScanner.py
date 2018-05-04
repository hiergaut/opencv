import cv2
import numpy as np
# import locale

# locale.setlocale(locale.LC_ALL, 'fr_FR')
# print(locale.getlocale())

# import track
# from fun import disp
# import fun

from PIL import Image
import pytesseract
# import imutils

# from matplotlib import pyplot as plt
import time

import track


global_debugMode = False


def resize(img):
    height, width = img.shape[:2]
    img = cv2.resize(img, (int(width / 2), int(height / 2)),
                     interpolation=cv2.INTER_CUBIC)
    return img


def threshold_byFirstWhite(img):
    # img =cv2.blur(img, (30, 30))
    # img = cv2.GaussianBlur(img, (21, 21), 0)
    # img = cv2.GaussianBlur(img, (21, 21), 0)
    # img = cv2.medianBlur(img, 21)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # search max value on histogram
    M = hist[0]
    i = 0
    for j in range(1, 256):
        cur = hist[j]
        if cur > M:
            M = cur
            i = j

    if i > 253:
        prev = hist[i]
    else:
        prev = hist[i] + hist[i + 1] + hist[i + 2]

    # search first croissant on the right
    for j in range(i + 15, 254):
        cur = hist[j] + hist[j + 1] + hist[j + 2]
        if cur >= prev:
            break
        prev = cur

    right = j

    if i < 2:
        prev = hist[i]
    else:
        prev = hist[i - 2] + hist[i - 1] + hist[i]

    # search first croissant on the left
    for j in range(i - 15, 2, -1):
        cur = hist[j - 2] + hist[j - 1] + hist[j]
        if cur >= prev:
            break
        prev = cur

    left = j

    # m = prev
    # pos = j - 50
    # for i in range(j - 1, j - 50, -1):
    #     if m > hist[i]:
    #         m = hist[i]
    #         pos = i

    # i = pos
    # gap =25

    # display histogram
    if global_debugMode:
        h = np.zeros((300, 256, 3))
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        hist = np.int32(np.around(hist))
        for x, y in enumerate(hist):
            cv2.line(h, (x, 0), (x, y), (255, 255, 255))

        # cv2.line(h, (i, 0), (i, 255), (0, 255, 0))
        cv2.line(h, (left, 0), (left, 255), (255, 0, 0))
        cv2.line(h, (right, 0), (right, 255), (0, 0, 255))
        y = np.flipud(h)

        cv2.imshow("y", y)
        cv2.moveWindow('y', 20, 20)

    # img = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU)[1]
    mask = cv2.threshold(img, right, 255, cv2.THRESH_BINARY)[1]
    # return img
    img = cv2.threshold(img, left, 255, cv2.THRESH_BINARY)[1]
    img = img - mask
    return img


def align(img_src):
    # print(global_debugMode)
    img = img_src.copy()

    img = threshold_byFirstWhite(img_src)
    # return img

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU)[1]
    # img = cv2.equalizeHist(img)

    # img = cv2.medianBlur(img, 3)
    # img =cv2.blur(img, (10, 10))
    # img = cv2.medianBlur(img, 3)
    # img = cv2.medianBlur(img, 3)
    img = cv2.medianBlur(img, 3)
    # img = cv2.medianBlur(img, 3)
    # return img_src
    # return img

    # img =cv2.Canny(img, 255, 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

    # cv2.imshow('c', img)
    # return img
    # img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, cnts, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # print("nb contours =" +str(len(cnts)))
    # disp(closed)

    if cnts == []:
        return img_src, None

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = img_src.copy()
    match = None
    # maxGap =50
    for cnt in cnts:

        l = len(cnt)
        # print(l)
        if 300 < l:

            approx = cv2.approxPolyDP(cnt, 0.02 * len(cnt), True)
            # approx = cv2.approxPolyDP(cnt, 2 * len(cnt), True)
            # print(approx)

            if len(approx) == 4:
                # [[p], [p2], [p3], [p4]] = approx
                # c =cv2.norm(p -p2)
                # c2 =cv2.norm(p2 -p3)
                # c3 =cv2.norm(p3 -p4)
                # c4 =cv2.norm(p4 -p)

                # print(abs(c -c3))
                # if abs(c -c3) < maxGap and abs(c2 -c4) < maxGap:
                match = approx.copy()

                cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)

                cpt = 1
                for p in approx:
                    # print(p)
                    # print(p[0][0])
                    # print(p[0][1])
                    cv2.putText(img, '_' + str(cpt), (p[0][0], p[0][1]),
                                font, 2, (255, 0, 0), 2, cv2.LINE_AA)
                    cpt += 1

                # else:
                #     cv2.drawContours(img, [cnt], 0, (127, 127, 255), 2)

                # break

            else:
                cv2.drawContours(img, [cnt], 0, (0, 0, 255), 2)

    # def dist(a, b):
    # print("cnts =" +str(len(cnts)))
    # (x, y), (x2, y2), _ = rect
    if global_debugMode:
        cv2.imshow('w', img)
    # cv2.moveWindow('w', 20, 20)

    # print(approx)
    # print(match)

    # return img
    # print(approx)
    if match is None:
        # print("bad quality")
        # return None
        return img, None

    assert(len(match) == 4)
    # return img

    # cv2.imwrite('last.jpg', img)

    # return img

    # return img

    # print(match)
    [[p], [p2], [p3], [p4]] = match

    zoom = 1
    w = zoom * int(cv2.norm(p - p2))
    h = zoom * int(cv2.norm(p - p4))
    if w > h:
        t = w
        w = h
        h = t

        pts = np.float32([[p4], [p], [p2], [p3]])
    else:
        pts = np.float32([[p], [p2], [p3], [p4]])

    # print(approx)
    pts2 = np.float32([[w, 0], [0, 0], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(pts, pts2)
    img2 = cv2.warpPerspective(img_src, M, (w, h))
    # pts = np.float32([])

    # l =cv2.Laplacian(img2, cv2.CV_64F)

    # blur = int(cv2.Sobel(img2, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1).var())
    # return l

    # blur =int(cv2.Laplacian(img2, cv2.CV_64F).var())
    # cv2.putText(img2, "blur : " +str(blur), (10, 30), font, 1, (0, blur /40, 255 -blur /40), 3)
    # img2 =cv2.Canny(img2, 127, 127)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.adaptiveThreshold(
    # img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # return img2
    # img2 =cv2.bitwise_not(img2)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    # img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel, iterations=3)
    # img2 = cv2.morphologyEx(img2, cv2.MORPH_TOPHAT, kernel, iterations=1)

    # print(blur)

    return img, img2


# def image_smoothening(img):
#     ret1, th1 = cv2.threshold(img, cv2.THRESH_BINARY, 255, cv2.THRESH_BINARY)
#     ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     blur = cv2.GaussianBlur(th2, (1, 1), 0)
#     ret3, th3 = cv2.threshold(
#         blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return th3


# def remove_noise_and_smooth(img):
#     # img = cv2.imread(file_name, 0)
#     filtered = cv2.adaptiveThreshold(img.astype(
#         np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
#     kernel = np.ones((1, 1), np.uint8)
#     opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
#     closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
#     img = image_smoothening(img)
#     or_image = cv2.bitwise_or(img, closing)
#     return or_image


def readText(img):
    h, w = img.shape[:2]
    # print(h, w)
    margin = 100
    img = img[margin:h - margin, margin: w - margin]

    # img = img[0:h, 0:500]

    # return img

    # img_orig =img.copy()

    # return img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # import track
    # track.binary(img)
    # cv2.GaussianBlur(img, (0, 0), 3)
    # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # img = cv2.filter2D(img, -1, kernel)
    # img =cv2.bilateralFilter(img, 9, 75, 75)
    # kernel = np.zeros( (9,9), np.float32)
    # kernel[4,4] = 2.0   #Identity, times two!

    # track.canny(img)
    # img = cv2.Canny(img, 127, 127)
    # #Create a box filter:
    # boxFilter = np.ones( (9,9), np.float32) / 81.0

    # #Subtract the two:
    # kernel = kernel - boxFilter

    # #Note that we are subject to overflow and underflow here...but I believe that
    # # filter2D clips top and bottom ranges on the output, plus you'd need a
    # # very bright or very dark pixel surrounded by the opposite type.

    # img = cv2.filter2D(img, -1, kernel)

    # remove_noise_and_smooth(img)
    # return img
    # track.binary(img)
    # img = cv2.adaptiveThreshold(
    # img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # _, mask = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
    # img = cv2.bitwise_and(img, img, mask=mask)


    # img =cv2.Laplacian(img, cv2.CV_64F)
    # img =cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    # return img
    # track.binary(img)
    # img = cv2.bitwise_not(img)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # # _, img = cv2.threshold(img, 87, 255, cv2.THRESH_BINARY)
    # n = np.zeros((3,3), np.uint8)
    # s = np.zeros((3,3), np.uint8)
    # w = np.zeros((3,3), np.uint8)
    # e = np.zeros((3,3), np.uint8)

    # n[0][1] = 1
    # s[2][1] = 1
    # w[1][0] = 1
    # e[1][2] = 1

    # img_n = cv2.erode(img, n, iterations=1)
    # img_s = cv2.erode(img, s, iterations=1)
    # img_w = cv2.erode(img, w, iterations=1)
    # img_e = cv2.erode(img, e, iterations=1)

    # img = img_n + img_s + img_w + img_e + img

    # return img

    # track.adaptThreshold(img)

    # print(np.ones((3, 3)))
    # kernel = np.ones((2, 2), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    # return img
    # cv::GaussianBlur(frame, image, cv::Size(0, 0), 3);
    # img2 =cv2.GaussianBlur(img, (0, 0), 3)
    # img = cv2.GaussianBlur(img, (21, 21), 0)
    # cv2.addWeighted(img, 1.5, img2, -0.5, 0, img2)
    # cv::addWeighted(frame, 1.5, image, -0.5, 0, image);

    # img =img2
    # track.adaptThreshold(img)
    # img = cv2.adaptiveThreshold(
        # img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 4)
        # img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 15)

    # img =cv2.medianBlur(img, 3)
    # img = cv2.bitwise_not(img)
    # kernel =cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # img =cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel, iterations=1)
    # return img
    # img =cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel, iterations=1)
    # img =cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
    # img =cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    # return img
    # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    #### GOLD BEST RECORD 38
    # img = cv2.bitwise_not(img)



    ## rotate
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
        # otherwise, just take the inverse of the angle to make
        # it positive
    else:
        angle = -angle

    angle +=0.2
    # angle +=-2

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h),
                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)



    # _,img = cv2.threshold(img,100,255,0)
    # _, img =cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    # img = cv2.imread('sofsk.png',0)
    # return img

    # img =cv2.medianBlur(img, 3)
    # img =cv2.medianBlur(img, 3)
    # return img

    # kernel =cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # img =cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # return img

    img2 = Image.fromarray(img)
    txt = pytesseract.image_to_string(img2, lang='fra')
    # txt = 'IATI de commencer cheval IATI Ajouter aueou   oeeeu o eoooooooouuu        oaau       eoau      aaaaeu             Ajouter'
    # print(txt)
    # exit(0)

    # keyword_list ='IATI TP1 Avant de commencer'.split()
    # keyword_list = ['motorcycle', 'bike', 'cycle', 'dirtbike', "long", 'Ajouter']
    file =open('motsFrancais.txt', 'r')
    keyword_list = file.read().split()
    print(len(keyword_list))
    # print(keyword_list)
    cpt =0
    # if set(keyword_list).intersection(txt.split()):
        # cpt +=1
    for word in txt.split():
        if word in keyword_list:
            print(word)
            cpt +=1
        # print("Found One")
    
    # print(len(txt.split()))
    nbWord =len(txt.split())
    if nbWord == 0:
        return img

    print("\naccuracy = ", cpt, '/', nbWord, ' ', "%.1f" % (cpt *100 /nbWord), "%")

    return img
    # return img
    # lisible =img

    # return img

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    # img = cv2.dilate(img, kernel, iterations=5)

    # return img
    # img = cv2.Canny(img, 127, 127)
    # cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # _, cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return img
    # img2 =img_orig.copy()
    # height, width = img.shape[:2]

    # cv2.imshow('w', img2)
    # cv2.waitKey(1)

    print(len(cnts))
    cpt = 0
    for cnt in cnts:

        [x, y, w, h] = cv2.boundingRect(cnt)
        if w > 20 and h > 10:
            print(cpt)
            cpt += 1
            if cpt > 2:
                break

            cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 255), 2)

            # print(x, y)
            # assert(w > 0)
            # assert(h > 0)
            # rect =img[x:x +w, y:y +h]
            rect = lisible[y:y + h, x:x + w]
            cv2.imshow('r', rect)
            cv2.moveWindow('r', 20, 20)

            # cv2.waitKey(1)
            # time.sleep(1)

            rect = Image.fromarray(rect)
            word = pytesseract.image_to_string(rect, lang='fra')
            print(word)
            cv2.putText(img2, word, (x + 5, y + h - 5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

            # return img2

    return img2

    # img =cv2.medianBlur(img, 1)
    img = cv2.bitwise_not(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=10)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img = cv2.erode(img, kernel)

    # disp(img)

    # img =cv2.imread('/home/gauthier/screen.png')
    img2 = Image.fromarray(img)
    txt = pytesseract.image_to_string(img2, lang='fra')
    print(txt)
    exit(0)

    return img


def test_withStream(stream):
    # DEBUG_MODE =True
    # camera =cv2.VideoCapture(0)
    # camera =cv2.VideoCapture('/dev/stdin')
    camera = cv2.VideoCapture(stream)

    while (True):
        _, frame = camera.read()

        img, img2 = align(frame)

        # if frame is not None:
        # cv2.imshow('frame', frame)
        # cv2.moveWindow('frame', 20, 20)
        cv2.imshow('img', img)

        if img2 is not None:
            cv2.imshow('img2', img2)

        if cv2.waitKey(40) == 27:
            # cv2.imwrite('last_camera.jpg', frame)
            break

    camera.release()
    cv2.destroyAllWindows()


def test_withPicture(pictureName):

    img = cv2.imread(pictureName)
    img = resize(img)

    img, img2 = align(img)

    cv2.imshow('img', img)

    if img2 is not None:
        cv2.imshow('img2', img2)

    while cv2.waitKey(100) != 27:
        continue
        # break


def run_readText(pictureName):
    img = cv2.imread(pictureName)
    img2, img = align(img)

    if img is None:
        cv2.imshow('img2', img2)
    else:
        img = readText(img)
        cv2.imshow('img', img)

    while cv2.waitKey(10) != 27:
        continue


# run_readText('full.jpg')
# exit(0)


import argparse
import sys
# from functools import partial

parser = argparse.ArgumentParser(
    description='scan paper like a printing scanner')
# parser.add_argument("-h", "--help", required=False, help="show this help message and exit")
parser.add_argument('-d', '--debug', help='use debug mode',
                    action='store_true')
parser.add_argument('-ap', '--picture', help='test align on PICTURE',
                    type=test_withPicture, metavar='PICTURE')
parser.add_argument('-as', '--stream', help='test align on STREAM',
                    type=test_withStream, metavar='STREAM')
parser.add_argument('-r', '--read', help='read text on picture',
                    type=run_readText, metavar='PICTURE')


if len(sys.argv) <= 1:
    sys.argv.append('--help')
# args =vars(parser.parse_args())
args = parser.parse_args()

if args.debug:
    global_debugMode = True
# options.func()


# test_withStream()
# test_withPicture()
