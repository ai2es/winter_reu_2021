cannydaynight.ipynb: compute contrast, Canny, and transmission scores for a time series of images. Save feature vectors with time stamps and ASOS readings to be used by an FCNN.

CCT_FCNN.ipynb: fit an FCNN to the contrast, Canny, transmission feature vectors

datasetfromASOS.ipynb: create a CNN dataset by assigning images to classes (folders) based on ASOS visibility readings.

eofs.ipynb: compute EOFs of raw grayscale images

MNetCoral.ipynb: attempt to use class ordinality to improve CNN accuracy

tfmobilenet.ipynb: building on the MobileNetV2 architecture for visibility classification. Can either perform transfer learning or random initialization.
