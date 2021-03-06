
import numpy as np
from union_find import UnionFind
import cv2
import edgeDetect 
import math

CONNECTIVITY_4 = 4
CONNECTIVITY_8 = 8


def connected_component_labelling(bool_input_image, connectivity_type=CONNECTIVITY_8):
	"""
		2 pass algorithm using disjoint-set data structure with Union-Find algorithms to maintain 
		record of label equivalences.

		Input: binary image as 2D boolean array.
		Output: 2D integer array of labelled pixels.

		1st pass: label image and record label equivalence classes.
		2nd pass: replace labels with their root labels.

		(optional 3rd pass: Flatten labels so they are consecutive integers starting from 1.)

	"""
	if connectivity_type !=4 and connectivity_type != 8:
		raise ValueError("Invalid connectivity type (choose 4 or 8)")


	image_width = len(bool_input_image[0])
	image_height = len(bool_input_image)

	# initialise efficient 2D int array with numpy
	# N.B. numpy matrix addressing syntax: array[y,x]
	labelled_image = np.zeros((image_height, image_width), dtype=np.int16)
	uf = UnionFind() # initialise union find data structure
	current_label = 1 # initialise label counter

	# 1st Pass: label image and record label equivalences
	for y, row in enumerate(bool_input_image):
		for x, pixel in enumerate(row):
			
			if pixel > 0:
				# Background pixel - leave output pixel value as 0
				pass
			else: 
				# Foreground pixel - work out what its label should be

				# Get set of neighbour's labels
				labels = neighbouring_labels(labelled_image, connectivity_type, x, y)

				if not labels:
					# If no neighbouring foreground pixels, new label -> use current_label 
					labelled_image[y,x] = current_label
					uf.MakeSet(current_label) # record label in disjoint set
					current_label = current_label + 1 # increment for next time				
				
				else:
					# Pixel is definitely part of a connected component: get smallest label of 
					# neighbours
					smallest_label = min(labels)
					labelled_image[y,x] = smallest_label

					if len(labels) > 1: # More than one type of label in component -> add 
										# equivalence class
						for label in labels:
							uf.Union(uf.GetNode(smallest_label), uf.GetNode(label))


	# 2nd Pass: replace labels with their root labels
	final_labels = {}
	new_label_number = 1

	for y, row in enumerate(labelled_image):
		for x, pixel_value in enumerate(row):
			
			if pixel_value > 0: # Foreground pixel
				# Get element's set's representative value and use as the pixel's new label
				new_label = uf.Find(uf.GetNode(pixel_value)).value 
				labelled_image[y,x] = new_label

				# Add label to list of labels used, for 3rd pass (flattening label list)
				if new_label not in final_labels:
					final_labels[new_label] = new_label_number
					new_label_number = new_label_number + 1


	# 3rd Pass: flatten label list so labels are consecutive integers starting from 1 (in order 
	# of top to bottom, left to right)
	# Different implementation of disjoint-set may remove the need for 3rd pass?
	for y, row in enumerate(labelled_image):
		for x, pixel_value in enumerate(row):
			
			if pixel_value > 0: # Foreground pixel
				labelled_image[y,x] = final_labels[pixel_value]

	return labelled_image



# Private functions ############################################################################
def neighbouring_labels(image, connectivity_type, x, y):
	"""
		Gets the set of neighbouring labels of pixel(x,y), depending on the connectivity type.

		Labelling kernel (only includes neighbouring pixels that have already been labelled - 
		row above and column to the left):

			Connectivity 4:
				    n
				 w  x  
			 
			Connectivity 8:
				nw  n  ne
				 w  x   
	"""

	labels = set()

	if (connectivity_type == CONNECTIVITY_4) or (connectivity_type == CONNECTIVITY_8):
		# West neighbour
		if x > 0: # Pixel is not on left edge of image
			west_neighbour = image[y,x-1]
			if west_neighbour > 0: # It's a labelled pixel
				labels.add(west_neighbour)

		# North neighbour
		if y > 0: # Pixel is not on top edge of image
			north_neighbour = image[y-1,x]
			if north_neighbour > 0: # It's a labelled pixel
				labels.add(north_neighbour)


		if connectivity_type == CONNECTIVITY_8:
			# North-West neighbour
			if x > 0 and y > 0: # pixel is not on left or top edges of image
				northwest_neighbour = image[y-1,x-1]
				if northwest_neighbour > 0: # it's a labelled pixel
					labels.add(northwest_neighbour)

			# North-East neighbour
			if y > 0 and x < len(image[y]) - 1: # Pixel is not on top or right edges of image
				northeast_neighbour = image[y-1,x+1]
				if northeast_neighbour > 0: # It's a labelled pixel
					labels.add(northeast_neighbour)
	else:
		print("Connectivity type not found.")

	return labels


def print_image(image):
	""" 
		Prints a 2D array nicely. For debugging.
	"""
	for y, row in enumerate(image):
		print(row)

def color(labels, numLabels):
	height = len(labels)
	width = len(labels[0])
	img_full= np.zeros((height,width,3), np.uint8)
	labelColors =[]
	numLabels= np.amax(labels)
	for i in range(numLabels):
		newColor = randomColors()
		labelColors += [newColor]
	for row in range(height):
		for col in range(width):
			if labels[row,col]>0:
				img_full[row,col] = labelColors[labels[row,col]-1]
			else:
				img_full[row,col] = [255,255,255]
	return img_full

numLabels = 0
labelColors = []

import random 
def randomColors():
    # brightness =  1.0 if bright else 0.7
    # hsv = [(i/N,1,brightness)for i in range(N)]
    # colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # random.shuffle(colors)
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    colors = [b,g,r]
    return colors

def skeletonize(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    img1 = img.copy()
    
    # Structuring Element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    # Create an empty output image to hold values
    thin = np.zeros(img.shape,dtype='uint8')
    
    # Loop until erosion leads to an empty set
    while (cv2.countNonZero(img1)!=0):
        # Erosion
        erode = cv2.erode(img1,kernel)
        # Opening on eroded image
        opening = cv2.morphologyEx(erode,cv2.MORPH_OPEN,kernel)
        # Subtract these two
        subset = erode - opening
        # Union of all previous sets
        thin = cv2.bitwise_or(subset,thin)
        # Set the eroded image for next iteration
        img1 = erode.copy()
    return thin



# Run from Terminal ############################################################################
if __name__ == "__main__":
	import sys
	if len(sys.argv) > 1: # At least 1 command line parameter
		image_path = str(sys.argv[1])
		if(len(sys.argv) > 2): # At least 2
			connectivity_type = int(sys.argv[2])
		else:
			connectivity_type = CONNECTIVITY_8
		image = cv2.imread(image_path,0)
		ret, bool_image = cv2.threshold(image, 65, 255, cv2.THRESH_BINARY)
		result = connected_component_labelling(bool_image, connectivity_type)
		my_set = set.union(*map(set,result))
		img_full = color(result,max(my_set))
		cv2.imwrite('res_'+str(image_path), img_full)
		flag = 0
		while(len(my_set)>200 and flag==0):
			kernel = np.ones((3,3),np.uint8)
			image = cv2.dilate(image,kernel,iterations = 2)
			ret, bool_image = cv2.threshold(image, 65, 255, cv2.THRESH_BINARY)
			result = connected_component_labelling(bool_image, connectivity_type)
			this_set = set.union(*map(set,result))
			if this_set == my_set:
				flag = 1
			my_set = this_set
			print(str(len(my_set)) + " components found")
		img_full = color(result,max(my_set))
		cv2.imwrite('eroded_'+str(image_path), img_full)
		img_edge = edgeDetect.edgeDetect(img_full)
		cv2.imwrite('edge_'+str(image_path), img_edge)
		img_skeleton = skeletonize(img_full)
		cv2.imwrite('skeleton_'+str(image_path),img_skeleton)
		
		bool_img = cv2.bitwise_not(bool_image)
		cnts, hierarchy = cv2.findContours(bool_img, cv2.RETR_EXTERNAL,
		                        cv2.CHAIN_APPROX_SIMPLE)
		
		# calculate x,y coordinate of center
		image_c =cv2.cvtColor(bool_image, cv2.COLOR_GRAY2BGR)
		# put text and highlight the center
		
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		image_c = cv2.circle(image_c, (int(x), int(y)), 5, (255, 255, 60), -1)
		cv2.putText(image_c, "centroid", (int(x) - 25, int(y) - 25),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		area = cv2.contourArea(c)
		perimeter =cv2.arcLength(c, True)
		# display the image
		M=cv2.moments(c)
		# Emin = min(M,key= M.get)
		# Emax = max(M,key=M.get )
		
		(x, y), (MA, ma), angle = cv2.fitEllipse(c)
		minmax = 4*math.pi*(area/(perimeter**2))
		# print(minmax)
		apratio = area//perimeter
		print("Area of Largest: Blob:", area)
		print("Perimeter of largest Blob:", perimeter)
		print("Coordinates of centroid:",x,y)
		print("circularity:", minmax)
		print("Angle:", angle)
		print("Area to perimeter ratio:", apratio)
		cv2.imwrite("centroid_"+image_path, image_c)
		




