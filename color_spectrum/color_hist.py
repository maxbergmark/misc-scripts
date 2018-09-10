import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as colors

def getHistogram(img):
	hsv = colors.rgb_to_hsv(img)
	hsv[:,:,2] /= 256
	# cv2.imshow('image', hsv[:,:,2])
	# cv2.waitKey(0)
	# quit()
	smooth_n = 10
	detail = 1
	smoother = np.array([2**(-detail*(i**2)) for i in range(-smooth_n, smooth_n+1)])
	smoother /= np.sum(smoother)
	ret = []

	for i in range(3):
		histr = cv2.calcHist([hsv],[i],None,[256],[0,1])
		yy = np.concatenate((histr[:,0], histr[:,0]))
		smoothed = np.convolve(smoother, yy)[smooth_n: len(histr) + smooth_n]
		# smoothed = histr
		# print(smoothed)
		s = np.sum(histr)
		smoothed *= 256/s
		ret.append(smoothed)
	return ret


def plotHistogram(histograms):
	f, axarr = plt.subplots(3, sharex=True)
	color = ('Hue','Saturation','Value')
	for i, smoothed in enumerate(histograms):
		count = 0

		# for y0, y1 in zip(smoothed[:-1], smoothed[1:]):
			# temp_color = [1,1,1-.3*(i==1)]
			# temp_color[i] = count/256
			# if (i == 2):
				# temp_color[1] = 0.3
			# axarr[i].plot([count, count+1], [y0, y1], color=colors.hsv_to_rgb(temp_color))
			# count += 1

		axarr[i].plot(smoothed,color = 'rgb'[i])

		axarr[i].set_xlim([0,255])
		axarr[i].set_ylim([0,np.max(smoothed)])
		axarr[i].set_title(color[i])

	plt.show()

# img = cv2.imread('icon.png')
# img = cv2.imread('4k-city.jpg')
# img = cv2.imread('color_balls.jpg')
# img = cv2.imread('beach.jpg')

# getHistogram(img)

cap = cv2.VideoCapture('kiwi.mkv')
while (cap.isOpened()):
	ret, frame = cap.read()
	# cv2.imshow('frame',frame)
	print("getting histogram")
	histograms = getHistogram(frame)
	print("plotting histogram")
	plotHistogram(histograms)
	# if cv2.waitKey(1) & 0xFF == ord('q'):
		# break

cap.release()
cv2.destroyAllWindows()