import cv2, numpy as np
from dict import *

def contourExtraction(im):

	dup_im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	# dup_im = cv2.GaussianBlur(dup_im, (5, 5), 0)
	thresh = cv2.adaptiveThreshold(dup_im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0)
	contours  = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]

	cv2.imwrite("out_adaptive_thresh.jpg", thresh)
	return contours

def polyApprox(contours):
	approx = []
	for contour in contours:
		epsilon = 0.05 * cv2.arcLength(contour, True)
		approx.append(cv2.approxPolyDP(contour, epsilon, True))

	return approx

def squareApprox(contours): 
	## Square Contours
	poly_contours = []
	for contour in contours: 
		if len(contour) ==  4:
			poly_contours.append(contour)

	return poly_contours

## Big Contours in terms of Area, removes the small noisy contours
def bigApprox(poly_contours, im):
	big_contours = []
	for contour in poly_contours:
		
		if  cv2.contourArea(contour) >= 0.001 * im.shape[0] * im.shape[1] :
			big_contours.append(contour)

	return big_contours

def getMarkers():
	## Converting Marker into a Binary Tree
	markers = []
	for i in range(1, 11):
		thresh = np.array(cv2.imread('markers/mark' + str(i) + '.jpg', cv2.CV_8UC1))
		# print thresh
		bin_string = []
		width = thresh.shape[1]
		height = thresh.shape[0]
		step_size = width/(num_bits + 4)
		i = j = 0
		

		while j + step_size <= height:
			i = 0
			word = ''
			while i + step_size <= width: 
				# print j, i
				block = thresh[j: j + step_size, i:i + step_size]
				centre_pixel = block[block.shape[0]/2][block.shape[1]/2]
				if centre_pixel == 255:
					word += "1"
				else:
					word += "0"
				i += step_size
			bin_string.append(word)
			j += step_size

		# cv2.imshow('image', thresh)
		# cv2.waitKey(0)
		bin_string = bin_string[2:-2]
		bin_string = [x[2:-2] for x in bin_string]
		markers.append(bin_string)
	return markers
	
	
def convToBin(thresh):
	## Converting Marker into a Binary Tree
	bin_string = []
	width = thresh.shape[1]
	height = thresh.shape[0]
	step_size = width/(num_bits + 2)
	i = j = 0
	

	while j + step_size <= height:
		i = 0
		word = ''
		while i + step_size <= width: 
			# print j, i
			block = thresh[j: j + step_size, i:i + step_size]
			centre_pixel = block[block.shape[0]/2][block.shape[1]/2]
			if centre_pixel == 255:
				word += "1"
			else:
				word += "0"
			i += step_size
		bin_string.append(word)
		j += step_size

	# cv2.imshow('image', thresh)
	# cv2.waitKey(0)
	return bin_string

def run(): 
	im = cv2.imread('test6.jpg')
	im = cv2.resize(im, None, fx = 0.25, fy = 0.25, interpolation = cv2.INTER_NEAREST)
	orig_im = im.copy()
	## Extract Contours 
	contours = contourExtraction(im)


	## Polygon Approximation
	contours = polyApprox(contours)
	# im2 = im.copy()
	# cv2.drawContours(im2, contours, -1, (255,0,0), 2)
	# cv2.imwrite("output_approx_poly.jpg", im2)

	## Square Approximation
	poly_contours = squareApprox(contours)

	
	## Big Contours Only
	big_contours = bigApprox(poly_contours, im)
	# print big_contours
	# return
	# im3 = im.copy()
	# cv2.drawContours(im3, big_contours, -1, (0,0,255), 2)
	# cv2.imwrite("output_approx_big.jpg", im3)
	c = []

	## saving image contours
	for contour in big_contours:
		# contour = big_contours[0]



		nw = 98
		nh = 98

		retval = cv2.getPerspectiveTransform(contour.astype(np.float32), np.array([[[0, 0], [nw, 0], [nw, nh], [0, nh]]]).astype(np.float32))
		persp_im = cv2.warpPerspective(im, retval, im.shape[0:2])
		# cv2.imwrite("output_persp.jpg", persp_im)
		# cv2.imshow('image', persp_im[0:100, 0:100])
		# cv2.waitKey(0)

		## Printing Grid Lines over detected marker
		persp_im = persp_im[0:nh, 0:nw]
		persp_im = cv2.resize(persp_im, None, fx = 2, fy = 2, interpolation = cv2.INTER_NEAREST)
		width = persp_im.shape[1]
		height = persp_im.shape[0]

		step_size = float(width/(num_bits + 2))
		i = 0.0
		while i < height:
			cv2.line(persp_im, (0, int(i)), (width, int(i)), (0, 255, 0)) 
			i += step_size

		step_size = float(height/(num_bits + 2))
		i = 0.0
		while i < width:
			cv2.line(persp_im, (int(i), 0), (int(i), height), (0, 255, 0)) 
			i += step_size


		## Adaptive Thresholding to get 
		blur = cv2.GaussianBlur(persp_im, (5, 5), 0)
		blur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
		ret3, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		global markers
		markers = [['11110', '10101', '00010', '01011', '11010'], ['00101', '00110', '11001', '00001', '01010'], ['11010', '10111', '01101', '01010', '10010'], ['01010', '01001', '10010', '01111', '00001'], ['01110', '01101', '10100', '01010', '11010'], ['00011', '00101', '01100', '11100', '11001'], ['01010', '11011', '00101', '10101', '00110'], ['11001', '11011', '10101', '10001', '01110'], ['10010', '01001', '01101', '00001', '11000'], ['00100', '10100', '01000', '11011', '11101']]
		marker = convToBin(thresh)
		marker = marker[1:-1]
		marker = [x[1:-1] for x in marker]


		if distDict(marker, markers) == 0:
			c.append(contour)
			

	cv2.drawContours(im, c, -1, (0,255,0), 2)
	cv2.imwrite("output.jpg", im)

	# ## Occlusion Detecion
	# cell_distr = []
	
	# i = 0
	# j = 0
	# width = orig_im.shape[1]
	# height = orig_im.shape[0]
	# step_size = width/15

	# while j + step_size <= height:
	# 	i = 0
	# 	while i + step_size <= width: 
	# 		block = persp_im[j: j + step_size, i:i + step_size]
	# 		# return block
	# 		em = cv2.ml.EM_create()
	# 		em.setClustersNumber(5)
	# 		block = block.astype(np.float32)
	# 		block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
	# 		em.trainEM(block)
			
	# 		i += step_size	
	# 	j += step_size



block = run()
# print getMarkers()
# markers = [['10110', '10111', '11011', '10100', '10001'], ['11011', '10101', '00110', '01000', '10010'], ['00110', '11101', '11001', '01010', '00111'], ['10100', '11011', '10101', '01101', '01000'], ['11110', '00011', '00101', '10010', '11011'], ['11001', '00100', '10101', '01110', '10111'], ['11101', '10100', '11000', '01010', '01001'], ['01001', '11000', '01000', '01010', '01100'], ['01010', '11010', '11101', '10100', '10000'], ['11110', '10101', '00010', '01011', '11010']]		
# sample = ['10110', '10111', '11011', '10100', '10001']
# sample = rotateBy90(sample)
# print distDict(sample, markers)
