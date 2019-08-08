import cv2
import numpy as np
from MtcnnDetector import FaceDetector

import caffe


frame_interval = 1

net_work_path = './model/2_deploy.prototxt'  #40*40
weight_path = './model/2_solver_iter_800000.caffemodel'
net = caffe.Net(net_work_path, weight_path, caffe.TEST)

def to_rgb(img):
  w, h = img.shape
  ret = np.empty((w, h, 3), dtype=np.uint8)
  ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
  return ret
  
  
def landmark(boxes):
	x11 = boxes[:,0]
	y11 = boxes[:,1]
	x22 = boxes[:,2]
	y22 = boxes[:,3]
	for i in range(x11.shape[0]):
		x1 = int(x11[i])
		y1 = int(y11[i])
		x2 = int(x22[i])
		y2 = int(y22[i])
		if x1 < 0: x1 = 0
		if y1 < 0: y1 = 0
		if x2 > frame.shape[1]: x2 = frame.shape[1]
		if y2 > frame.shape[0]: y2 = frame.shape[0]
		#cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
		roi = frame[y1:y2 + 1, x1:x2 + 1, ]
		gary_frame = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
		#w = 60
		#h = 60
		w = 40
		h = 40	
		#print (image)
		res = cv2.resize(gary_frame, (w, h), 0.0, 0.0, interpolation=cv2.INTER_CUBIC)
		resize_mat = np.float32(res)

		m = np.zeros((w, h))
		sd = np.zeros((w, h))
		mean, std_dev = cv2.meanStdDev(resize_mat, m, sd)
		new_m = mean[0][0]
		new_sd = std_dev[0][0]
		new_frame = (resize_mat - new_m) / (0.000001 + new_sd)

		if new_frame.shape[0] != net.blobs['data'].data[0].shape or new_frame.shape[1] != net.blobs['data'].data[1].shape:
			print ("Incorrect , resize to correct dimensions.")

		net.blobs['data'].data[...] = new_frame

		out = net.forward()

		#points = net.blobs['Dense3'].data[0].flatten()
		points = net.blobs['Dense2'].data[0].flatten()
		
		point_pair_l = len(points)
		for i in range(point_pair_l // 2):
			x = points[2*i] * (x2 - x1) + x1
			y = points[2*i+1] * (y2 - y1) + y1
			cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), 2)
		
	return frame  

if __name__ == '__main__':   
 
 
	detector = FaceDetector(minsize = 60, gpuid = 0, fastresize = False)
	videoCapture = cv2.VideoCapture('1234.mp4')
	#videoCapture = cv.CaptureFromCAM(0)	
	sucess,frame = videoCapture.read()        
	c=0 
	while True:
		# Capture frame-by-frame
		ret, frame = videoCapture.read()
		#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		##print(frame.shape)  
		timeF = frame_interval

		if(c%timeF == 0): #frame_interval==3, face detection every 3 frames
        
			find_results=[]
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			if gray.ndim == 2:
				img = to_rgb(gray)

			 
			total_boxes,points,numbox = detector.detectface(frame)
			#print(total_boxes)

			if len(total_boxes) != 0:
				frame = landmark(total_boxes)
			
			for i in range(numbox):
			    cv2.rectangle(frame,(int(total_boxes[i][0]),int(total_boxes[i][1])),(int(total_boxes[i][2]),int(total_boxes[i][3])),(0,255,0),2)        
			    #for j in range(5):        
			        #cv2.circle(frame,(int(points[j,i]),int(points[j+5,i])),2,(0,0,255),2)
				
		c = c + 1		
		cv2.imshow('Video', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break