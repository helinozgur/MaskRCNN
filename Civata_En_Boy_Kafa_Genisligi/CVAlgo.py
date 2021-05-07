ret, bw_img = cv.threshold(gri, 127, 255, cv.THRESH_BINARY_INV)
kernel = np.ones((1, int(round(length_w))), np.uint8)
angle_dgree = (180 / np.pi) * angle
print('AAANNGGLLEEE:' + str(angle_dgree))
rotated = ndimage.rotate(kernel, angle_dgree, reshape=False)
opening = cv.morphologyEx(bw, cv.MORPH_OPEN, rotated)
plt.imshow(opening)