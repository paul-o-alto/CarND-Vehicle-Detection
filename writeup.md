**Vehicle Detection Project**

The goals / steps of this project were the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, one can also apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run a pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###Histogram of Oriented Gradients (HOG)

####1. Extracting HOG features from the training images.

The code for this step is contained in the function `get_hog_features` in the file called `tracking.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Settling on the final choice of HOG parameters.

I am still experimenting here.

####3. Training a classifier using selected HOG features (and, optionally, color features).

I trained a linear SVM using HOG features extracted from a grayscale image. I also appended to those features a color histogram based on the YCrCb color space (converted from BGR) and a spatial histogram. These can be found in the functions `bin_spatial`, `channel_hist`, `get_hog_features`, and `extract_features` in the file called `tracking.py`

###Sliding Window Search

####1. Implementing a sliding window search.  Scales to search and how much to overlap windows?

I am still trying to find the write balance of window sizes. Generally I have been trying size of 100, 200, 300. I have not found that larger window sizes are helpful (more false positives). I do this in the `pipeline` function of `tracking.py`. I extract features from the windows I recieve from the function `slide_window`.

![alt text][image3]

####2. Some examples of test images to demonstrate how the pipeline is working. Optimizing the performance of your classifier.

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided an ok result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. A link to the final video output.
Here's a [link to my video result](./project_video_out.mp4)


####2. Implementing a filter for false positives and a method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Problems / issues faced in the implementation of this project.  Where will the pipeline likely fail?  What could be done to make it more robust?

I am really having an issue with false positives (for example, trees and highway dividers). I also have spurious true detections of cars on the opposite side of the freeway, but yet I sometimes have trouble consistently picking up cars on the same side of the road.  

