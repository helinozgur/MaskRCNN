import cv2 as cv
import matplotlib.pyplot as plt
import deneme_functions
import numpy as np
import imutils
from scipy.spatial import distance as dist
from scipy import ndimage

# Load image
src = cv.imread('./images/20210201_121113.jpg')
# Convert image to grayscale
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# Convert image to binary
_, bw = cv.threshold(gray, 150, 250, cv.THRESH_BINARY | cv.THRESH_OTSU)
# Find all the contours in the thresholded image
contours, _ = cv.findContours(bw, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
hull = []
localIndex = 0

for i, c in enumerate(contours):
    # Calculate the area of each contour
    area = cv.contourArea(c)
    # Ignore contours that are too small or too large
    if area < 1e4 or 1e6 < area:
        continue
    img = np.zeros(gray.shape, np.uint8)
    contour_screw = cv.drawContours(img, [c], -1, (255, 255, 255), -1)
    angle, w2, h2, cntr, angle_line = deneme_functions.getOrientation(c, img)
    print("Width Coordinates:\n" + "--------------------------------------")
    co1, co2, img_w, length_w, array_w, btw_w = deneme_functions.getCoordinates(contour_screw, w2, gray)
    print("Height Coordinates:\n" + "--------------------------------------")
    co_h_1, co_h_2, img_h, length_h, array_h, btw_h = deneme_functions.getCoordinates(contour_screw, h2, gray)
    print("Height of the bolt:" + str(round(length_h, 3)))
    print("Width of the bolt:" + str(round(length_w, 3)))

    # #Visualization of result##
    img = np.zeros(src.shape, np.uint8)
    contour = cv.drawContours(img, [c], -1, (255, 255, 255), -1)
    _, w_2 = deneme_functions.getOrientation(c, src)
    btw = cv.bitwise_and(contour, w_2)
    btw_or = cv.bitwise_or(btw, src)
    cntr_h = ((cntr[0] + 100), (cntr[1] - 50))
    cv.putText(btw_or, "height:" + str(round(length_h, 3)) + "px", cntr_h, cv.FONT_HERSHEY_SIMPLEX,
               1, (0, 0, 255), 2, cv.LINE_AA)
    cntr_w = ((cntr[0] + 100), (cntr[1] - 100))
    cv.putText(btw_or, "width:" + str(round(length_w, 3)) + "px", cntr_w, cv.FONT_HERSHEY_SIMPLEX,
               1, (0, 0, 255), 2, cv.LINE_AA)
    #### FINDING THE HEAD SIZE OF THE BOLT ####
    kernel = np.ones((3, (int(length_w) + int(length_w*0.2))), np.uint8)
    angle_line_dgree = (180 / np.pi) * angle_line
    result=180-angle_line_dgree
    print("Angle:" + str(result))
    rotated = ndimage.rotate(kernel, result, reshape=True)
    opening = cv.morphologyEx(contour_screw, cv.MORPH_OPEN, rotated)
    plt.imshow(opening)
    plt.show()
    cnts, _ = cv.findContours(opening, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for a, cn in enumerate(cnts):
        area = cv.contourArea(cn)
        if area > 1e6 or area < 1e3:
            continue
        box = cv.minAreaRect(cn)
        box = cv.boxPoints(box) if imutils.is_cv2() else cv.boxPoints(box)
        box = np.array(box, dtype="int")
        cv.drawContours(opening, [box], 0, (255, 0, 0), 2)


        def midpoint(ptA, ptB):
            return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        (c1,c2)= midpoint((int(tltrX), int(tltrY)),(int(blbrX), int(blbrY)))
        cntr_hs=((int(c1) + 100), (int(c2) +10))
        if dA < dB:
            cv.line(btw_or, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                    (255, 0, 0), 2)
            print("Head Size:" + str(round(dA, 3)) + "px")
            #cv.circle(btw_or,(int(c1),int(c2)), 3, (255, 0, 255), 3)
            cv.putText(btw_or, "head size:" + str(round(dA, 3)) + "px", cntr_hs, cv.FONT_HERSHEY_SIMPLEX,
                       1, (255, 0, 0), 2, cv.LINE_AA)
        elif dB < dA:
            cv.line(btw_or, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                    (255, 0, 0), 2)
            print("Head Size:" + str(round(dB, 3)) + "px")
            #cv.circle(btw_or, (int(c1), int(c2)), 3, (255, 0, 255), 3)
            cv.putText(btw_or, "head size:" + str(round(dB, 3)) + "px", cntr_hs, cv.FONT_HERSHEY_SIMPLEX,
                       1, (255, 0, 0), 2, cv.LINE_AA)
        plt.imshow(btw_or)
        plt.show()

