import numpy as np
import cv2
import glob
import time
import os
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
from scipy.ndimage.measurements import label
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split # >= 0.18
from sklearn.cross_validation import train_test_split
from moviepy.editor import VideoFileClip

DEBUG = not True
MODEL = None
SCALER = None
MODEL_FILE  = 'svm.pkl'
SCALER_FILE = 'scaler.pkl' 
HEATMAP = None
HEATMAP_THRESH = 2

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    # defining a 3 channel or 1 channel color to fill 
    # the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], (0,0,255), thick)
    # Return the image copy with boxes drawn
    return draw_img

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def add_heat(bbox_list):
    global HEATMAP

    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        print("Adding heat for %s" % (box,))
        HEATMAP[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        #HEATMAP[box[0][0]:box[1][0], box[0][1]:box[1][1]] += 1

def apply_threshold():
    global HEATMAP

    # Zero out pixels below the threshold
    HEATMAP[HEATMAP <= HEATMAP_THRESH] = 0 # <=
    # To prevent a runaway heatmap over multiple frames
    #HEATMAP[HEATMAP > HEATMAP_THRESH] -= 1 # >
    #HEATMAP[HEATMAP == threshold] = 1

def ground_heatmap():
    HEATMAP[HEATMAP > HEATMAP_THRESH] = HEATMAP_THRESH

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None: x_start_stop[0] = 0
    if x_start_stop[1] == None: x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None: y_start_stop[0] = 0
    if y_start_stop[1] == None: y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# TEMPLATE MATCHING, PROBABLY NOT USEFUL DIRECTLY
# Define a function to search for template matches
# and return a list of bounding boxes
def find_matches(img, template_list):
    # Define an empty list to take bbox coords
    bbox_list = []
    # Define matching method
    # Other options include: cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCORR',
    #         'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'
    method = cv2.TM_CCOEFF_NORMED
    # Iterate through template list
    for temp in template_list:
        # Read in templates one by one
        tmp = cv2.imread(temp)
        # Use cv2.matchTemplate() to search the image
        result = cv2.matchTemplate(img, tmp, method)
        # Use cv2.minMaxLoc() to extract the location of the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # Determine a bounding box for the match
        w, h = (tmp.shape[1], tmp.shape[0])
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        # Append bbox position to list
        bbox_list.append((top_left, bottom_right))
        # Return the list of bounding boxes
        
    return bbox_list

# Define a function to compute color histogram features  
def channel_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    c1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    c2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    c3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = c1_hist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((c1_hist[0], c2_hist[0], c3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features, c1_hist, c2_hist, c3_hist, bin_centers

# Define a function to compute color histogram features  
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
def bin_spatial(img, size=(32, 32)):
    feature_image = np.copy(img)             
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel() 
    # Return the feature vector
    return features


# Constants specific to hog
ORIENT = 9
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2

# Define a function to return HOG features and visualization
def get_hog_features(img, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = \
            hog(img, 
                orientations=ORIENT, 
                pixels_per_cell=(PIX_PER_CELL, PIX_PER_CELL),
                cells_per_block=(CELL_PER_BLOCK, CELL_PER_BLOCK), 
                visualise=True) #, feature_vector=False)
        if DEBUG: plt.imshow(hog_image); plt.show()
        return features
    else:      
        features = hog(img, 
                       orientations=ORIENT, 
                       pixels_per_cell=(PIX_PER_CELL, PIX_PER_CELL),
                       cells_per_block=(CELL_PER_BLOCK, CELL_PER_BLOCK), 
                       visualise=False) #, feature_vector=feature_vec)
        return features

def extract_features(imgs, hog_plus=False,
                     spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        if len(imgs) > 1:
            image = cv2.imread(file)
        else:
            image = file
        
        if image.shape[1] > 64 or image.shape[0] > 64:
            image = cv2.resize(image, (64, 64), 
                               fx=64/image.shape[1], fy=64/image.shape[0])

        nongray_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog_features = get_hog_features(feature_image,
                                         vis=False, feature_vec=True)

        spatial_features = []
        hist_features = []
        if hog_plus:
            # Apply bin_spatial() to get spatial color features
            if spatial_size: 
                spatial_features = bin_spatial(nongray_image, size=spatial_size)
            # Apply channel_hist() also with a color space option now
            if hist_bins and hist_range: 
                hist_features = channel_hist(nongray_image, 
                                       nbins=hist_bins, 
                                       bins_range=hist_range)
                hist_features = hist_features[0]
        
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))

    # Return list of feature vectors
    return features

# Define a function to return some characteristics of the dataset 
def data_look(car_list, notcar_list):

    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = cv2.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict

def train_svm(scaled_X, y):

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(t2-t, 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Train Accuracy of SVC = ', svc.score(X_train, y_train))
    print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
    # Check the prediction time for a single sample
    t=time.time()
    prediction = svc.predict(X_test[0].reshape(1, -1))
    t2 = time.time()
    print(t2-t, 'Seconds to predict with SVC')

    return svc

def get_training_specs(cars, notcars):
    
    print('# cars: %s, # not-cars: %s' % (len(cars), len(notcars)))
    car_features    = extract_features(cars)
    notcar_features = extract_features(notcars)
    print('Car features: %s, Not-cars features: %s'
          % (len(car_features), len(notcar_features)))

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
 
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    return X_scaler, X, y


def pipeline(img):
    global HEATMAP, MODEL, SCALER

    # This part is optional (helps in avoiding searching the sky, for example)
    img_size = img.shape[0:2]
    img_size = img_size[::-1] # Reverse order
    # HSV/HLS might be better for object detection
    # because of saturation channel
    # img = cv2.cvtColor(img, COLOR_CONVERSION)
    #vertices = np.float32([[0, img_size[1]/2],
    #                       [0, img_size[1]], 
    #                       [img_size[0], img_size[1]],
    #                       [img_size[0], img_size[1]/2]])
    #img = region_of_interest(img, vertices)

   
    # Try different window sizes
    for size in [256, 128]:
        window_list = slide_window(img, 
                               x_start_stop=[0, img_size[0]], 
                               y_start_stop=[img_size[1]/2, img_size[1]],
                               xy_window=(size,size)) # SMALL TO LARGE!
        bbox_list = []
        # ...and use the trained classifer to search for vehicles in subimages
        for window in window_list:
            sub_img = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
            #sub_img = img[window[0][0]:window[1][0], window[0][1]:window[1][1]]
            img_features = extract_features([sub_img])
            # Apply the scaler to X
            scaled_X = SCALER.transform(img_features)
            prediction = MODEL.predict(scaled_X)
            if prediction == 1:
                if DEBUG: print('Found car!')
                bbox_list.append(window)
        add_heat(bbox_list)
    
    if DEBUG: plt.imshow(HEATMAP); plt.show()
 
    # Run your pipeline on a video stream and create a heat map of recurring 
    # detections frame by frame to reject outliers and follow detected vehicles.
    apply_threshold()
    #if DEBUG: plt.imshow(HEATMAP); plt.show()
    labels = label(HEATMAP)
    if DEBUG and labels: print(labels[1], 'cars found')
    #if DEBUG: plt.imshow(labels[0], cmap='gray'); plt.show()

    # Draw bounding boxes on a copy of the image
    out_img = draw_labeled_bboxes(np.copy(img), labels)
    if DEBUG: plt.imshow(out_img); plt.show()
    ground_heatmap()

    return out_img 

def main():
    global MODEL, SCALER, HEATMAP

    try:
        MODEL = joblib.load(MODEL_FILE)
        SCALER = joblib.load(SCALER_FILE)
    except Exception as e:
        print('Got exception %s when trying to load model file' % e)

    # Divide up into cars and notcars
    images = glob.glob('./training_set/*/*/*.jpeg')
    cars = []
    notcars = []
    for image in images:
        if 'image' in image or 'extra' in image:
            notcars.append(image)
        else:
            cars.append(image)

    if not MODEL or not SCALER:
        SCALER, X, y = get_training_specs(cars, notcars)
        # Apply the scaler to X
        scaled_X = SCALER.transform(X)
        print("Training a Linear SVM classifier")
        MODEL = train_svm(scaled_X, y)
        joblib.dump(MODEL, MODEL_FILE)
        joblib.dump(SCALER, SCALER_FILE)


    # y, x
    HEATMAP = np.zeros((720,1280,3)).astype(np.float)

    if DEBUG:
        images = glob.glob('test_images/test*.jpg')
        print("Looking at images %s" % images)
        for idx, fname in enumerate(images):
            image = cv2.imread(fname)
            pipeline(image) 
    else:
        # For processing video
        output_file = 'out.mp4'
        clip = VideoFileClip('project_video.mp4')
        out_clip = clip.fl_image(pipeline)
        out_clip.write_videofile(output_file, audio=False)

if __name__ == '__main__':
    main()
