from dict import *
import cv2
import numpy as np

# markers = genDictionary()
# print markers
# markers = [['10110', '10111', '11011', '10100', '10001'], ['11011', '10101', '00110', '01000', '10010'], ['00110', '11101', '11001', '01010', '00111'], ['10100', '11011', '10101', '01101', '01000'], ['11110', '00011', '00101', '10010', '11011'], ['11001', '00100', '10101', '01110', '10111'], ['11101', '10100', '11000', '01010', '01001'], ['01001', '11000', '01000', '01010', '01100'], ['01010', '11010', '11101', '10100', '10000'], ['11110', '10101', '00010', '01011', '11010']]

i = 1
for marker in markers:

	im_marker = convToLol(marker)
	im_marker = np.array(im_marker).astype(np.uint8)
	im_marker = im_marker * 255
	im_marker = np.lib.pad(im_marker, (1,1), 'constant', constant_values=(0, 0))
	im_marker = np.lib.pad(im_marker, (1,1), 'constant', constant_values=(255, 255))

	res = cv2.resize(im_marker, None, fx = 50, fy = 50, interpolation = cv2.INTER_NEAREST)
	cv2.imwrite("markers/mark" + str(i) + ".jpg", res)
	print markers
	# f = fopen("markers_data/mark" + str(i) +".txt", "w")
	# for row in res: 

	# i += 1
