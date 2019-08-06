# import the necessary packages
import numpy as np
import argparse
import cv2
 
# Using the ArgumentParser to load the input image file
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image file")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])

## print(image.shape) -- (1446, 1268, 3)

# Converting BGR image to Grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# Flipping the background and foreground
gray = cv2.bitwise_not(gray)
# The pixels greater than 0 will be assigned 255 
thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

## print(thresh.shape) -- (1446, 1268)

coords = np.column_stack(np.where(thresh > 0))

## print(coords.shape) -- (108540, 2)
## print(coords) -- ((722.6476440429688, 633.48583984375), (1094.2032470703125, 696.0149536132812), -24.010149002075195)

angle = cv2.minAreaRect(coords)[-1]

## print(angle)  -24.010149002075195

if angle < -45:
	angle = -(90 + angle)

else :
	angle = -angle

# Transforming the image
(h,w) = image.shape[:2]
center = (w//2,h//2)
## It returns a 2*3 matrix consisting of values derived from alpha and beta
## alpha = scale * cos(angle)
## beta = scale * sine(angle)
matrix = cv2.getRotationMatrix2D(center,angle,1.0)

## src – input image and dst – output image 
## The function warpAffine transforms the source image using the specified matrix:
## dst(x,y) = src(M11X + M12Y + M13, M21X + M22Y + M23)
## when the flag WARP_INVERSE_MAP is set. 
## Otherwise, the transformation is first inverted with invertAffineTransform() and then put in the formula above instead of M . 
rotated = cv2.warpAffine(image, matrix, (w,h), flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE)

# Displaying the image
cv2.putText(rotated, "Rotation Angle : {:.2f} degrees".format(angle),
	(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

print("[INFO] angle: {:.3f} " .format(angle))
cv2.imshow("Input Image",image)
cv2.imshow("Rotated Image",rotated)
cv2.waitKey(0)


