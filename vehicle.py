import numpy as np

class Vehicle():
    """
    This class keeps track of existing detections. An instance of
    this class is not necessarily always the same vehicle. Sometimes
    an existing Vehicle instances will come to represent a different
    actual vehicle. Instances of this class should be placed in an 
    iterator which will be, effectively, a cache of previous 
    detections. When averages are recomputed in this class, the new  
    bounding box should be checked for an actual detection on  
    the image in question.
    """

    def __init__(self):
        self.n = 30 # Roughly 1 second of video
        self.detected = False # was it detected in the last frame
        
        self.num_detections = 0 # number of times this vehicle has been
        self.num_nodetections = 0 # number of consecutive non-detection
        
        self.xpixels = None # pixel x values of last detection
        self.ypixels = None # pixel y values of last detection
        
        self.recent_xfitted = [] # x position of last n fits
        self.bestx = None # average x of last n fits
        self.recent_yfitted = [] # y position of last n fits
        self.besty = None # average y of last n fits
        
        self.recent_wfitted = [] # width of last n fits
        self.bestw = None # average width of last n fits
        self.recent_hfitted = [] # height of last n fits
        self.besth = None # average height of last n fits
    
    def get_avg_bboxes(self):
        """
        Returns both a small and large bbox centered around the same
        x, y coordinates
        """
        
        # This coercion to int doesn't need to be percise
        half_height = int(self.besth/2)
        half_width  = int(self.bestw/2)
        
        # Remember: y grows downward
        small = ((self.bestx - half_width, self.besty - half_height),
                 (self.bestx + half_width, self.besty + half_height))
        large = ((self.bestx - self.bestw, self.besty - self.besth),
                 (self.bestx + self.bestw, self.besty + self.besth))
                 
        return (small, large)
    
    def set_detection(self, outcome):
        """
        This is key. Once you move the centroid via averaging,
        you still need to make sure the centroid is indeed valid.
        We increment num_detections and num_nodetections. If the
        latter becomes greater than the former, we didn't really
        detect anything.
        """
        if outcome == 1:
            self.detected = True
            self.num_detections += 1
        else:
            self.detected = False
            self.num_nodetections += 1
    
    def is_valid(self):
        # Basically, our confidence over time
        valid =  self.num_detections > self.num_nodetections
        return valid
    
    def set_pixels(self, nonzerox, nonzeroy):
        self.xpixels = nonzerox
        self.ypixels = nonzeroy
    
    def append_xy(self, bbox):
        self.recent_xfitted.append((bbox[0][0] + bbox[1][0])/2)
        self.recent_yfitted.append((bbox[0][1] + bbox[1][1])/2)
        
        if len(self.recent_xfitted) > self.n:
            self.recent_xfitted = self.recent_xfitted[1:]
        if len(self.recent_yfitted) > self.n:
            self.recent_yfitted = self.recent_yfitted[1:]
       
        # This coercion to int doesn't need to be percise
        self.bestx = int(np.mean(self.recent_xfitted))
        self.besty = int(np.mean(self.recent_yfitted))
        
        # Absolute value to handle errors associated mainly
        # with the inverted y values in cv2 images
        self.recent_wfitted.append(abs(bbox[1][0] - bbox[0][0]))
        self.recent_hfitted.append(abs(bbox[1][1] - bbox[0][1]))
        
        if len(self.recent_wfitted) > self.n:
            self.recent_wfitted = self.recent_wfitted[1:]
        if len(self.recent_hfitted) > self.n:
            self.recent_hfitted = self.recent_hfitted[1:]
        
        # This coercion to int doesn't need to be percise
        self.bestw = int(np.mean(self.recent_wfitted))
        self.besth = int(np.mean(self.recent_hfitted))
        
