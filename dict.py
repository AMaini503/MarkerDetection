import random, time, cv2
import numpy as np
from math import floor, ceil

num_bits = 5
size = num_bits
markers = []
binary_patterns = []
# words = []
dictSize = 10

def convToLol(m1):
	new = []
	for word in m1:
		[a, b, c, d, e] = word
		new.append([a, b, c, d, e])
	return new

def Hamming(m1, m2):
	sum = 0
	# print m1, m2
	for i in range(len(m1)):
		sum += bin( int(m1[i], 2) ^ int(m2[i], 2)).count("1")
	return sum

def rotateBy90(m1):
	# m1 = ['10100', '01001', '00001', '11110', '11101']
	m1 = convToLol(m1)
	m1 = np.array(m1).transpose()
	m1 = np.fliplr(m1)
	m1 = m1.tolist()
	m2 = []
	for word in m1:
		m2.append(''.join(word))
	return m2

def rotateK(m1, k):
	while k > 0:
		m1 = rotateBy90(m1)
		k -= 1
	return m1

def distance(m1, m2):
	min_sum = num_bits * num_bits + 1
	for k in range(4): 
		curr_sum = Hamming(m1, rotateK(m2, k))
		if min_sum > curr_sum:
			min_sum = curr_sum
	return min_sum

def distDict(m1, markers):
	min_sum = num_bits * num_bits + 1
	for m2 in markers:
		curr_sum = distance(m1, m2)
		if min_sum > curr_sum:
			min_sum = curr_sum
	return min_sum

def selfDist(m1):
	min_sum = num_bits + 1
	for k in xrange(1, 4):
		curr_sum = Hamming(m1, rotateK(m1, k))
		if min_sum > curr_sum:
			min_sum = curr_sum
	return min_sum

def T(bin_string):
	count = 0
	for i in range(len(bin_string)- 1):
		if bin_string[i] != bin_string[i+1]: 
			count += 1
	return count/float(num_bits - 1)

def O(bin_string):
	count  = 0
	if len(markers) == 0:
		return 1.0
	for marker in markers:
		for word in marker:
			if bin_string == word:
				count += 1
	return 1 - count/float(num_bits * len(markers))

def P(bin_string, binary_patterns):
	norm = sum([T(word) * O(word)  for word in binary_patterns])
	return T(bin_string) * O(bin_string) / float(norm)

def init(): 
	
	global binary_patterns

	words = []
	decimal_patterns = range(0, 2 ** num_bits)
	format_string = '{0:0' + str(num_bits) + 'b}'
	
	binary_patterns = [format_string.format(num) for num in decimal_patterns]
	
	for word in binary_patterns:
		words.append( (word, P(word, binary_patterns)))
	return words
# print words

def genRandomMarker(words):
	# words.sort(key=lambda tup: tup[1])
	
	words2 = []
	words2.append(words[0])
	for i in range(1, len(words)):
		words2.append((words[i][0], words[i][1] + words2[i - 1][1]))
	# print words2
	marker = []

	ind = 0 
	# return 0
	while len(marker) != size :
		random.seed(time.time())
		x = random.random()
		# print x
		for i in range(0, len(words2)):
			if x < words2[i][1] :
				ind = i
				break
		if words2[ind][0] not in marker:
			marker.append(words2[ind][0])
	# markers.append(marker)
	return marker

def update(words, markers):
	for i in range(len(words)):
		# print binary_patterns
		words[i] = (words[i][0], P(words[i][0], binary_patterns))

def genDictionary():

	global markers
	words = init()

	## Init threshold
	C = floor( num_bits ** 2 / 4.0)
	threshold = 2 * floor(4 * C / 3.0)

	## Number Unproductive Iterations
	num_unp_iter = 0
	max_unp_iter = 10
	
	while len(markers) != dictSize :
		m = genRandomMarker(words)
	
		if selfDist(m) >= threshold and distDict(m, markers) >= threshold:
			markers.append(m)
			
			num_unp_iter = 0
			update(words, markers)
		else :
			num_unp_iter += 1
			if num_unp_iter == max_unp_iter:
				threshold -= 1
				num_unp_iter = 0

	return markers