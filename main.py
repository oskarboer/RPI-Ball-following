import cv2
import numpy as np
from time import sleep

vid = cv2.VideoCapture(0) 

# Creating mask for red:

while(True): 
	  

	ret, frame = vid.read()

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

	closeness_coefisient = np.sum(mask) / np.sum(np.full(mask.shape, 255))
	
	moments = cv2.moments(mask) # doCUmEnTatiOn: https://en.wikipedia.org/wiki/Image_moment https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#moments
	centroid_x = moments["m10"] / moments["m00"]
	centroid_y = moments["m01"] / moments["m00"]

	
	print(int(centroid_x), int(centroid_y), closeness_coefisient)
	
	cv2.imshow('frame', erosion)
	
	
	

	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break
  
# After the loop release the cap object 
vid.release()

cv2.destroyAllWindows() 
