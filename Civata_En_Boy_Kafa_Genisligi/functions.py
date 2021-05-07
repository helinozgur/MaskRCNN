import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from math import atan2, cos, sin, sqrt, pi
def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    line=cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 5, -1)

    # # create the arrow hooks
    # p[0] = q[0] + 9 * cos(angle + pi / 4)
    # p[1] = q[1] + 9 * sin(angle + pi / 4)
    # #cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 5, cv.LINE_AA)
    #
    # p[0] = q[0] + 9 * cos(angle - pi / 4)
    # p[1] = q[1] + 9 * sin(angle - pi / 4)
    # #cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 5, cv.LINE_AA)
    return line

def getOrientation(pts, img):
    # Construct a buffer used by the pca analysis+
    size = len(pts)
    data_pts = np.empty((size, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)

    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))

    # Draw the principal components
    cv.circle(img, cntr, 3, (255, 0, 255), 10)
    h = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    w = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    contour_line = np.zeros(img.shape, np.uint8)
    h1 = drawAxis(contour_line, cntr, h, (0, 255, 0), 1)
    w1 = drawAxis(h1, cntr, w, (255, 255, 0), 1)
    h2 = drawAxis(w1, cntr, h, (0, 255, 0), -1)
    w2 = drawAxis(h2, cntr, w, (255, 255, 0), -1)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    return angle,w2
