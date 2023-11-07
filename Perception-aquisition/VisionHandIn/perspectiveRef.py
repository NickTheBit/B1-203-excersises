# Just to use a root image for desired perspective
import cv2
import numpy as np

class reference:
	corners = []
	img = ""
	ret = ""

	def __init__(self):
		# Load the image
		self.img = cv2.imread("rawImages/Guill_aligned.jpg")

		# Scaling Down the image 1 times specifying a single scale factor.
		scale_down = 0.3
		self.img = cv2.resize(self.img, None, fx=scale_down, fy=scale_down, interpolation=cv2.INTER_LINEAR)

		# Displaying chess-board features
		self.ret, self.corners = cv2.findChessboardCorners(self.img, (4, 6),
		                                              flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
		                                                    cv2.CALIB_CB_FAST_CHECK +
		                                                    cv2.CALIB_CB_NORMALIZE_IMAGE)
		if not self.ret:
			print("We ain't found shit!")

	def get_ref_corners(self):
		return self.corners

	def show_ref_image(self):
		# Drawing and printing the overlay of the checkerboard.
		print("Corners of checkerboard")
		print(self.corners)
		print(self.corners.shape)
		fnl = cv2.drawChessboardCorners(self.img, (4, 6), self.corners, self.ret)
		cv2.imshow("Reference Alignment", fnl)
		cv2.waitKey(0)
