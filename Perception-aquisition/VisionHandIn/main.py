import cv2
import numpy as np

# Load the image
img = cv2.imread("rawImages/Guill_misaligned.jpg")

# Scaling Down the image 1 times specifying a single scale factor.
scale_down = 0.3
img = cv2.resize(img, None, fx=scale_down, fy=scale_down, interpolation=cv2.INTER_LINEAR)

# Just for drawing a rectangle if needed.
# cv2.rectangle(img, (378, 575), (600, 733), (35, 35, 99), 2)

# Displaying chess-board features
ret, corners = cv2.findChessboardCorners(img, (4, 6),
                                         flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                               cv2.CALIB_CB_FAST_CHECK +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
if not ret:
    print("We ain't found shit!")

print(corners)
# Drawing and printing the overlay of the checkerboard.
# print("Corners of checkerboard")
# print(corners)
# print(corners.shape)
# fnl = cv2.drawChessboardCorners(img, (4, 6), corners, ret)
# cv2.imshow("fnl", fnl)
# cv2.waitKey(0)

# # Trying to correct
ref_point_raw = np.float32([corners[0][0], corners[3][0], corners[23][0]])
ideal_points = np.float32(
    [[corners[0][0][0], corners[0][0][1]], [corners[3][0][0], corners[0][0][0]], [corners[23][0][0], corners[0][0][1]]])

# Transformations
opa = cv2.getAffineTransform(ref_point_raw, ideal_points)
cols, rows, _ = img.shape
transformed_img = cv2.warpAffine(img, opa, (cols, rows))
cv2.imshow("gay", transformed_img)
cv2.waitKey(0)
