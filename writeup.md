**Vehicle Detection Project**

The goals / steps of this project were the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run a pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.jpg
[image3]: ./output_images/sliding_windows.jpg
[image41]: ./output_images/all_pos_bboxes5.jpg
[image42]: ./output_images/all_pos_bboxes6.jpg
[image43]: ./output_images/all_pos_bboxes7.jpg
[image44]: ./output_images/all_pos_bboxes8.jpg
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[image8]: ./output_images/HOG_frame1_ch0.png
[image9]: ./output_images/HOG_frame1_ch1.png
[image10]: ./output_images/HOG_frame1_ch2.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###Histogram of Oriented Gradients (HOG)

####1. Extracting HOG features from the training images.

The code for this step is contained in the function `get_hog_features` in the file called `pipeline.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HLS` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image8]
![alt text][image9]
![alt text][image10]

####2. Settling on the final choice of HOG parameters.

I started with the default HOG parameters (orientation=9, pixels_per_cell=8x8, cells_per_block=2x2) . I mainly tweaked the orientation bin count. For pixels per cell, I started with 8x8 and moved up to 16x16. This greatly improved the performance compared to 8x8. I kept cells per block as 2x2 so I would get a decent amount of block normalization.

####3. Training a classifier using selected HOG features (and, optionally, color features).

I trained a linear SVM using HOG features extracted from an HLS image (on each channel seperately). I also appended to those features a color histogram based on the HLS color space (converted from BGR) and a spatial histogram. These can be found in the functions `bin_spatial`, `channel_hist`, `get_hog_features`, and `extract_features` in the file called `pipeline.py`

###Sliding Window Search

####1. Implementing a sliding window search.  Scales to search and how much to overlap windows?

I settles on one single high level window size of 128. I have not found that larger window sizes are helpful (more false positives). I take each window of 128x128 and I find smaller windows within this window (of size 64x64). I use the detections in these smaller windows as votes of confidence (to add to the heatmap). The amount I add to the heatmap is dictated by the number of confidence votes in each larger window from these smaller window detections. I do this in the `pipeline` function of `pipeline.py`. I extract features from the windows I recieve from the function `slide_window`.

![alt text][image3]

####2. Some examples of test images to demonstrate how the pipeline is working. Optimizing the performance of your classifier.

Ultimately I searched on two scales using HLS 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided an ok result.  Here are some example images:

![alt text][image41]
![alt text][image42]
![alt text][image43]
![alt text][image44]
---

### Video Implementation

####1. A link to the final video output.
Here's a [link to my video result](./project_video_out.mp4)


####2. Implementing a filter for false positives and a method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. In addition to this, I kept a cache of these heatmap labels. This was saved as a dictionary between frames and used to check for previous detections (before moving onto the "less-informed" sliding window approach). Each previous detection was represented as an instance of the `Vehicle` class located in `vehicle.py`. 

This was more of a heuristic because the dictionary keys were based on the label numbers from the heatmap label function. Because of this, the cache of the previous detections was used as a list of -suggested- locations to search. If the detection was no longer valid (the SVM said 'no car') then no bounding box was drawn based on the previous detection.

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

I really had an issue with false positives (for example, trees and highway dividers). I also had spurious true detections of cars on the opposite side of the freeway, but yet I sometimes had trouble consistently picking up cars on the same side of the road. It wasn't until I added the Vehicle class, and the cache of previous detections, that I finally saw smoother bounding boxes between frames. This use of suggested search locations was highly useful in finding the same vehicle again. 

 

