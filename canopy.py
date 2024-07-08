import cv2            
import matplotlib.pyplot as plt 
import time
# save started time
start_time = time.time()

#import image
image1 = cv2.imread('IMG_20211001_152035_MP.jpg')
#denoise the image
new_image = cv2.fastNlMeansDenoisingColored(image1,None,10,10,7,3)
blue_channel = new_image[:,:,2] # Select Blue channel

#Thresholding image using otsu method
ret, otsu = cv2.threshold(blue_channel, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  
#Thresholding image using local thresholding method
local_thresh = cv2.adaptiveThreshold(blue_channel, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 1) 
   
#merge both thresholded images
result = (cv2.add(255-otsu, 255-local_thresh))-255
flat_result=result.flatten()

#count the canopy pixels
count=0
for i in flat_result:
    if i <1:
        count+=1
#calculate percentage and print
print("Canopy percentage: " , round((count/len(flat_result))*100),"%")

#plot the images
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.tight_layout()
plt.subplot(2, 2, 1)
plt.imshow(new_image)
plt.title('Denoised image')
plt.axis("off")
plt.subplot(2, 2, 2)
plt.imshow(blue_channel,cmap='gray')
plt.title('Blue channel image')
plt.axis("off")
plt.subplot(2, 2, 3)
plt.imshow(otsu,cmap='gray')
plt.title('otsu image')
plt.axis("off")
plt.subplot(2, 2, 4)
plt.imshow(local_thresh,cmap='gray')
plt.title('Local thresholded image')
plt.axis("off")
plt.show()

plt.imshow(result,cmap='gray')
plt.title('Merged image')
plt.axis("off")
plt.show()

#print execution time
print("--- %s seconds ---" % round((time.time() - start_time),3))