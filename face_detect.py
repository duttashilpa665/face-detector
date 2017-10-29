import cv2
import sys

# Get user supplied values
# imagePath = sys.argv[1]
image_names =sys.argv[1:]
if len(image_names)>0:
	image_found=1
else:
	image_found=0
cascPath = "haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)
cam = cv2.VideoCapture(0)
# Read the image
i=0
while(1):
	if i==len(image_names) and image_found==1:
		break
	if image_found==0:
		retval, image = cam.read()
	else: 
		image = cv2.imread(str(image_names[i]))
		i+=1
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
	    gray,
	    scaleFactor=1.1,
	    minNeighbors=5,
	    minSize=(30, 30),
	    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
	)

	print("Found {0} faces!".format(len(faces)))

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
	    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
	if image_found:
		cv2.imshow("result from "+str(image_names[i-1]),image)
	else:	
		cv2.imshow("Faces found", image)
	cv2.waitKey(30)
cv2.waitKey(0)