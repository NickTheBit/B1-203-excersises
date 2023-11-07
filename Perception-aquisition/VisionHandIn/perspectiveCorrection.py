import cv2
import numpy as np
import perspectiveRef as perspective_reference
import glob

# Get the checkerboard from a properly aligned image
pref = perspective_reference.reference().get_ref_corners()

def allign(img, ref_corners):
	# Scaling Down the image 1 times specifying a single scale factor.
	scale_down = 0.3
	img = cv2.resize(img, None, fx=scale_down, fy=scale_down, interpolation=cv2.INTER_LINEAR)

	# Displaying chess-board features
	ret, corners = cv2.findChessboardCorners(img, (4, 6),
	                                         flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
	                                               cv2.CALIB_CB_FAST_CHECK +
	                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
	if not ret:
		print("We ain't found shit!")

	ref_point_raw = np.array([corners[0], corners[15], corners[20]])
	ideal_points = np.array([ref_corners[0], ref_corners[15], ref_corners[20]])

	# Transformations
	opa = cv2.getAffineTransform(ref_point_raw, ideal_points)
	cols, rows, _ = img.shape
	transformed_img = cv2.warpAffine(img, opa, (cols, rows))
	cv2.imshow("Fixed perspective", transformed_img)
	cv2.waitKey(0)


# find and adjust all images
for name in glob.glob("rawImages/*"):
	print(name)
	img = cv2.imread(name)
	try:
		allign(img, pref)
	except:
		print("fail")
