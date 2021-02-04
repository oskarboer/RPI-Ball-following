import cv2
import numpy as np
from time import sleep

class motor:
	def __init__(self, pin_a, pin_b):
		self.pin_a
		self.pin_b


class pid:
	def __init__(self, P, I, D):
		self.P = P
		self.I = I
		self.D = D
	





vid = cv2.VideoCapture(0)

# specific size needs to be set up depending on camera resolution
full_mask = np.sum(np.full((480, 640), 255)) 

while(True): 
	ret, frame = vid.read()
	
	#frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) #not so imortant

	img_hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	lower = np.array([165,25,0])
	upper = np.array([179,255,255])
	mask = cv2.inRange(img_hsv, lower, upper)
	# result = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)

	# ret, threshold = cv2.threshold(mask,200,255,cv2.THRESH_BINARY)
	
	# Erosion and dilation with cv2 to remove most of the noise:
	kernel = np.ones((5,5),np.uint8)
	erosion = cv2.erode(mask, kernel, iterations = 1)
	
	# kernel_3 = np.ones((5,5),np.uint8)
	# dilation = cv2.dilate(img_third,kernel_3,iterations = 1)

	closeness_coefisient = np.sum(mask) / full_mask #/ np.sum(np.full(mask.shape, 255))
	
	if (closeness_coefisient < 0.10): #magick number
		# doCUmEnTatiOn: https://en.wikipedia.org/wiki/Image_moment
		# https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#moments
		moments = cv2.moments(mask) 
		if moments["m00"] != 0:
			centroid_x = moments["m10"] / moments["m00"]
			centroid_y = moments["m01"] / moments["m00"]
		# we are only interested in y coordinate
		# 	(because it is physically rotated on the robot)
		# 	closeness is needed to judge the distanse to the object
		print(int(centroid_y), closeness_coefisient)
		# _________________________
		# PID_SPACE
		
		
		
		
		
		
		# _________________________
	
#	cv2.imshow('frame', erosion)
	
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break
  
# After the loop release the cap object 
vid.release()

cv2.destroyAllWindows() 
