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
from sklearn.utils import shuffle
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
def channel_hist(img, nbins=32):
    # Compute the histogram of the RGB channels separately
    c1_hist = np.histogram(img[:,:,0], bins=nbins)
    c2_hist = np.histogram(img[:,:,1], bins=nbins)
    c3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Generating bin centers
    bin_edges = c1_hist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((c1_hist[0], c2_hist[0], c3_hist[0]))
    #print(hist_features)
    # Return the individual histograms, bin_centers and feature vector
    #return hist_features, c1_hist, c2_hist, c3_hist, bin_centers
    # This is how it was in the lecture code
    return c1_hist, c2_hist, c3_hist, bin_centers, hist_features
    
# Define a function to compute color histogram features  
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
# Size was 32x32 before, but that is probably way to big
#    It will generate a feature_vec 3072 long! 16x16 is 
#    only 256, much more reasonable
# NOTE: All three channels? hstack them?
def bin_spatial(img, size=(32, 32)):
    feature_image = np.copy(img)             
    # Use cv2.resize().ravel() to create the feature vector
    features_ch1 = cv2.resize(feature_image[:,:,0], size).ravel() 
    features_ch2 = cv2.resize(feature_image[:,:,1], size).ravel()
    features_ch3 = cv2.resize(feature_image[:,:,2], size).ravel()
    # Return the feature vector
    return np.hstack((features_ch1, features_ch2, features_ch3))


# Constants specific to hog
ORIENT = 9 # Usually, 6 to 12
PIX_PER_CELL = 8 #(8,8)
CELL_PER_BLOCK = 2 #(2,2)
# Will be 7x7x2x2x9 long

# Define a function to return HOG features and visualization
# MAYBE ADD BACK transform_sqrt param?
def get_hog_features(img, vis=False, feature_vec=False):
    # feature_vec induces behavior similar to .ravel()
    hog_image = None
    if vis == True:
        features, hog_image = \
            hog(img, 
                orientations=ORIENT, 
                pixels_per_cell=(PIX_PER_CELL, PIX_PER_CELL),
                cells_per_block=(CELL_PER_BLOCK, CELL_PER_BLOCK), 
                visualise=True)#, feature_vector=feature_vec)
    else:      
        features = hog(img, 
                       orientations=ORIENT, 
                       pixels_per_cell=(PIX_PER_CELL, PIX_PER_CELL),
                       cells_per_block=(CELL_PER_BLOCK, CELL_PER_BLOCK), 
                       visualise=False)#, feature_vector=feature_vec)
    #if DEBUG: cv2.imwrite('./output_images/HOG_example.jpg', hog_image)
    return features, hog_image

def extract_features(imgs, include_hog=True, include_spatial=True, include_color_hist=True,
                     gray_hog=False, #True,
                     spatial_size=(32, 32), hist_bins=16):
    # Create a list to append feature vectors to
    set_features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        if len(imgs) > 1:
            image = cv2.imread(file)
        else:
            image = file
       
        t_size = 64 
        if image.shape[1] > t_size or image.shape[0] > t_size:
            image = cv2.resize(image, (t_size, t_size), 
                               fx=t_size/image.shape[1], fy=t_size/image.shape[0])

        all_features = ([],)
        if include_hog:
            if gray_hog:
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        
                tot_hog_features, _ = get_hog_features(feature_image, 
                                        vis=DEBUG, feature_vec=False)
            else:
                color_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
                hog_channels = []
                for channel in [0,1,2]:
                    feature_image = color_image[:,:,channel]
                    hog_features, _ = get_hog_features(feature_image,
                                             vis=DEBUG, feature_vec=False)
                    hog_channels.append(hog_features)
                tot_hog_features = np.ravel(hog_channels)

            all_features += (tot_hog_features,)
        color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        if include_spatial:
            # Apply bin_spatial() to get spatial color features
            if spatial_size: 
                spatial_features = bin_spatial(color_image, size=spatial_size)
                if spatial_features is not None:
                    #print("Got spatial features!")
                    all_features += (spatial_features,)
            # Apply channel_hist() also with a color space option now
        if include_color_hist:
            hist_features = channel_hist(color_image, 
                                         nbins=hist_bins)
            hist_features = hist_features[4]
            if hist_features is not None:
                #print("Got channel histogram!")
                all_features += (hist_features,)
       
        img_features = np.concatenate(all_features)
        if len(imgs) == 1: 
            return img_features
        else:
            set_features.append(img_features)

    # Return list of feature vectors
    return set_features

# Define a function to return some characteristics of the dataset 
def data_look(car_list, notcar_list):

    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img_car = cv2.imread(car_list[1])
    example_img_notcar = cv2.imread(notcar_list[1])
    vis = np.concatenate((example_img_car, example_img_notcar), axis=1)
    cv2.imwrite('./output_images/car_not_car.png', vis)
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img_car.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img_car.dtype
    # Return data_dict
    return data_dict

def train_svm(car_X, car_y, notcar_X, notcar_y):

    print("Training on %s features" % len(car_X[0])) # 8,000 if fine
    
    scaled_car_X = SCALER.transform(car_X)
    scaled_notcar_X = SCALER.transform(notcar_X)
    # Split up data into randomized training and test sets

    rand_state = np.random.randint(0, 100)
    X_car_train, X_car_test, y_car_train, y_car_test = \
	train_test_split(scaled_car_X, car_y, 
                         test_size=0.2, random_state=rand_state)
    X_notcar_train, X_notcar_test, y_notcar_train, y_notcar_test = \
        train_test_split(scaled_notcar_X, notcar_y,
                         test_size=0.2, random_state=rand_state)

    train_rs = np.random.randint(0,100) # Same for both X and y
    X_train = shuffle(np.concatenate((X_car_train, X_notcar_train)),
                      random_state=train_rs)
    y_train = shuffle(np.concatenate((y_car_train, y_notcar_train)),
                      random_state=train_rs)

    test_rs = np.random.randint(0,100) # Same for both X and y
    X_test = shuffle(np.concatenate((X_car_test, X_notcar_test)),
                     random_state=test_rs)
    y_test = shuffle(np.concatenate((y_car_test, y_notcar_test)),
                     random_state=test_rs)

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
    global SCALER   
 
    print(data_look(cars, notcars))
    print('# cars: %s, # not-cars: %s' % (len(cars), len(notcars)))
    car_features    = extract_features(cars)
    notcar_features = extract_features(notcars)
    print('Car features: %s, Not-cars features: %s'
          % (len(car_features), len(notcar_features)))
    print('Example Car Features: %s' % (car_features[0],))
    print('Example non-Car Features: %s' % (notcar_features[0],))
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Fit a per-column scaler
    SCALER = StandardScaler().fit(X)
    car_labels = np.ones(len(car_features))
    notcar_labels = np.zeros(len(notcar_features))
    return car_features, car_labels, notcar_features, notcar_labels

def add_heat(bbox_list, heatmap=None):
    global HEATMAP

    if heatmap is None:
        heatmap = HEATMAP

    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    return heatmap

def pipeline(img):
    global HEATMAP, MODEL, SCALER
    img_size = img.shape[0:2]
    img_size = img_size[::-1] # Reverse order

    # Try different window sizes
    bbox_list = []
    # Example 64x64 image, 8x8 block, 7x7  x2x2x9
    for size, overlap, y_start, y_stop in zip([256,512], # NOTE: Multiple of HOG cell size!
                                              [0.5]*2, 
                                              [360,380],
                                              [656]*2):
        #if DEBUG: print("size=%s, overlap=%s, y_stop=%s" % (size, overlap, y_stop))
        window_list = slide_window(img, 
                          x_start_stop=[0, img_size[0]], 
                          y_start_stop=[y_start, y_stop], # S to L
                          xy_window=(size,size), 
                          xy_overlap=(overlap, overlap)
                          )
        # Here we have the macro windows that we will iterate over
        for window in window_list:
            sub_img = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
            sub_img_size = sub_img.shape[0:2][::-1]
            img_features = extract_features([sub_img], 
                                              include_hog=True,
                                              include_spatial=True, 
                                              include_color_hist=True)
            # Apply the scaler to X
            scaled_X = SCALER.transform(np.array(img_features).reshape(1, -1))
            prediction = MODEL.predict(scaled_X)
            if prediction[0] == 1:
            #if prediction == 1:
                bbox_list.append(window)
                print("Found car!") 
        
    add_heat(bbox_list)
    
    # Run your pipeline on a video stream and create a heat map of recurring 
    # detections frame by frame to reject outliers and follow detected vehicles.
    labels = label(HEATMAP)

    if DEBUG: 
        final_map = np.clip(HEATMAP - 2, 0, 255)
        #plt.imshow(final_map, cmap='hot'); plt.show()
        print(labels[1], 'cars found')
        cv2.imwrite('./output_images/labels_map.png', labels[0])

    if labels[1] > 0:
        # Draw bounding boxes on a copy of the image
        out_img = draw_labeled_bboxes(np.copy(img), labels)
    else:
        out_img = img

    HEATMAP[(HEATMAP <= HEATMAP_THRESH)] = 0

    return out_img 

def main():
    global DEBUG, MODEL, SCALER, HEATMAP


    try:
        MODEL = joblib.load(MODEL_FILE)
        SCALER = joblib.load(SCALER_FILE)
    except Exception as e:
        print('Got exception %s when trying to load model file' % e)

    # Divide up into cars and notcars
    print("Using KITTI-only car dataset, to avoid time series issues.")
    cars    = glob.glob('./training_set/vehicles/*/*.png') #KITTI* 
    notcars = glob.glob('./training_set/non-vehicles/*/*.png') #Extras

    if not MODEL or not SCALER:
        car_X, car_y, notcar_X, notcar_y = \
            get_training_specs(cars, notcars)
        print("Training a Linear SVM classifier")
        MODEL = train_svm(car_X, car_y, notcar_X, notcar_y)
        joblib.dump(MODEL, MODEL_FILE)
        joblib.dump(SCALER, SCALER_FILE)

    HEATMAP = np.zeros((720,1280)).astype(np.uint8)

    if DEBUG:
        images = glob.glob('test_images/test*.jpg')
        print("Looking at images %s" % images)
        for idx, fname in enumerate(images):
            image = cv2.imread(fname)
            result = pipeline(image) 
            out_fn = './output_images/%s' % fname.split('/')[-1]
            print('Saving output to %s' % out_fn)
            cv2.imwrite(out_fn, result)
            # Need to zero out, because these are 'independent' images
            PREV_HEATMAP = np.zeros_like(image[:,:,0]).astype(np.uint8) 
            #DEBUG = False
   
        output_file = 'test_video_out.mp4'
        clip = VideoFileClip('test_video.mp4')
        out_clip = clip.fl_image(pipeline)
        out_clip.write_videofile(output_file, audio=False)
 
    else:   
        # For processing video
        output_file = 'project_video_out.mp4'
        clip = VideoFileClip('project_video.mp4')
        out_clip = clip.fl_image(pipeline)
        out_clip.write_videofile(output_file, audio=False)

if __name__ == '__main__':
    main()
