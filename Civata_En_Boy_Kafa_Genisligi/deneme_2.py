import cv2 as cv
import deneme_functions
import numpy as np
import matplotlib.pyplot as plt
cap = cv.VideoCapture(0)
fps = cap.get(cv.cv.CV_CAP_PROP_FPS)

while (True):

    # Capture frames in the video
    ret, frame = cap.read()

    # describe the type of font
    # to be used.
    font = cv.FONT_HERSHEY_SIMPLEX
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Convert image to binary
    _, bw = cv.threshold(gray, 50, 230, cv.THRESH_BINARY | cv.THRESH_OTSU)
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
        _, w2, h2, cntr = deneme_functions.getOrientation(c, img)
        print("Width Coordinates:\n" + "--------------------------------------")
        co1, co2, img_w, length_w = deneme_functions.getCoordinates(contour_screw, w2, gray)
        print("Height Coordinates:\n" + "--------------------------------------")
        co_h_1, co_h_2, img_h, length_h = deneme_functions.getCoordinates(contour_screw, h2, gray)
        print("Height of the bolt:" + str(round(length_h, 3)))
        print("Width of the bolt:" + str(round(length_w, 3)))

        # #Visualization of result##
        img = np.zeros(frame.shape, np.uint8)
        contour = cv.drawContours(img, [c], -1, (255, 255, 255), -1)
        _, w_2 = deneme_functions.getOrientation(c, frame)
        btw = cv.bitwise_and(contour, w_2)
        btw_or = cv.bitwise_or(btw, frame)
        cntr_h = ((cntr[0] + 50), (cntr[1] - 50))
        cv.putText(btw_or, "height:" + str(round(length_h, 3)) + "px", cntr_h, cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 2, cv.LINE_AA)
        cntr_w = ((cntr[0] + 100), (cntr[1] - 100))
        cv.putText(btw_or, "width:" + str(round(length_w, 3)) + "px", cntr_w, cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 2, cv.LINE_AA)
    # Use putText() method for
    # inserting text on video
    cv.putText(frame,
                'TEXT ON VIDEO',
                (50, 50),
                font, 1,
                (0, 255, 255),
                2,
                cv.LINE_4)

    # Display the resulting frame
    cv.imshow('video', frame)

    # creating 'q' as the quit
    # button for the video
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# release the cap object
cap.release()
# close all windows
cv.destroyAllWindows()