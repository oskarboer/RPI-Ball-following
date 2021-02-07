import cv2
import numpy as np
from time import sleep
import RPi.GPIO as GPIO
import math


GPIO.setmode(GPIO.BOARD)

PWM_FREQ = 50

class motor:
	def __init__(self, pin_a: int, pin_b: int):
		self.pin_a = pin_a
		self.pin_b = pin_b
		GPIO.setup(self.pin_a, GPIO.OUT)
		GPIO.setup(self.pin_b, GPIO.OUT)
		self.A = GPIO.PWM(self.pin_a, PWM_FREQ)
		self.B = GPIO.PWM(self.pin_b, PWM_FREQ)
	
	def start(self):
		self.A.start(0)
		self.B.start(0)
		
	def run(self, speed):
		speed = min(max(int(speed), -100), 100)
		if speed > 0:
			self.A.ChangeDutyCycle(abs(speed))
			self.B.ChangeDutyCycle(0)
		else:
			self.A.ChangeDutyCycle(0)
			self.B.ChangeDutyCycle(abs(speed))

class pid:
	def __init__(self, P, I, D):
		self.P = P
		self.I = I
		self.D = D
		self.target = 0
		self.integral = 0
		self.derivative = 0
		self.previous_error = 0
		
	def cycle(self, feedback):
		error = self.target - feedback
		self.integral += error
		self.derivative = error - self.previous_error
		
		correction = self.P * error + self.I * self.integral + self.D * self.derivative
		
		self.previous_error = error
		
		return correction
	
	def set_target(self, new_target):
		self.target = new_target
			
class robot:
	def __init__(self, motorR: motor, motorL: motor):
		self.motorR = motorR
		self.motorL = motorL
	
	def stop(self):
		self.motorR.run(0)
		self.motorL.run(0)


vid = cv2.VideoCapture(0)

# specific size needs to be set up depending on camera resolution
full_mask = np.sum(np.full((480, 640), 255)) 

# _________________________
# Initialisation
motorR = motor(40, 38)
motorL = motor(35, 37)
robot = robot(motorR, motorL)
distance_pid = pid(2.5, 0.0, 0.1)
distance_pid.set_target(55)

rotation_pid = pid(0.10, 0.0, 0.0)
rotation_pid.set_target(240)

motorR.start();
motorL.start();

while(True): 
	ret, frame = vid.read()
	
	#frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) #not so imortant

	img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	lower = np.array([165,25,0])
	upper = np.array([179,255,255])
	mask = cv2.inRange(img_hsv, lower, upper)
	# result = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)

	# ret, threshold = cv2.threshold(mask,200,255,cv2.THRESH_BINARY)
	
	# Erosion and dilation with cv2 to remove most of the noise:
	kernel = np.ones((5,5),np.uint8)
	erosion = cv2.erode(mask, kernel, iterations = 2)
	
	# kernel_3 = np.ones((5,5),np.uint8)
	# dilation = cv2.dilate(img_third,kernel_3,iterations = 1)

	closeness_coefisient = (np.sum(mask) / full_mask)**0.5 #/ np.sum(np.full(mask.shape, 255))
	
	# closeness_coefisient = closeness_coefisient if closeness_coefisient > 0.02 else 0
	# print(closeness_coefisient**0.5)
	
	if (closeness_coefisient > 0.15): #magick number
		# doCUmEnTatiOn: https://en.wikipedia.org/wiki/Image_moment
		# https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#moments
		moments = cv2.moments(mask) 
		if moments["m00"] != 0:
			centroid_x = moments["m10"] / moments["m00"]
			centroid_y = moments["m01"] / moments["m00"]
		# we are only interested in y coordinate
		# 	(because it is physically rotated on the robot)
		# 	closeness is needed to judge the distanse to the object
		# print(int(centroid_y), closeness_coefisient)
		
		
		# _________________________
		# PID_SPACE
		
		
		distance_speed_correction = distance_pid.cycle(closeness_coefisient * 100)
		
		speedR = distance_speed_correction
		speedL = distance_speed_correction
		
		
		rotation_speed_correction = rotation_pid.cycle(centroid_y)
		
		speedR -= rotation_speed_correction
		speedL += rotation_speed_correction
		
		
		print(rotation_speed_correction)
		
		motorR.run(speedR)
		motorL.run(speedL)
		
		
		# _________________________
	else:
		robot.stop()
#	cv2.imshow('frame', erosion)
	
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break
  
# After the loop release the cap object 
vid.release()

cv2.destroyAllWindows() 
