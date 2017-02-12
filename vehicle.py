class Vehicle():
    """
    This class keeps track of existing detections. An instance of
    this class is not necessarily always the same vehicle. Sometimes
    an existing Vehicle instances will come to represent a different
    actual vehicle. Instances of this class should be placed in an 
    iterator which will be, effectively, a cache of previous 
    detections. When averages are recomputed in this class, the new  
    bounding box should be check for an actual detection on the     
    image in question.
    """

    def __init__(self):
        self.n = 30 # Roughly 1 second of video
        self.detected = False # was it detected in the last frame
        
        self.n_detections = 0 # number of times this vehicle has been
        self.n_nodetections = 0 # number of consecutive non-detection
        
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
    
    def get_avg_bbox(self):
        half_height = self.recent_hfitted/2
        half_width  = self.recent_wfitted/2
        return ((self.bestx - half_width, self.besty - half_height),
                (self.bestx + half_width, self.besty + half.height))
    
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
       
        self.bestx = np.mean(self.recent_xfitted)
        self.besty = np.mean(self.recent_yfitted)
        
        self.recent_wfitted.append(abs(bbox[1][0] - bbox[0][0]))
        self.recent_hfitted.append(abs(bbox[1][1] - bbox[0][1]))
        
        if len(self.recent_wfitted) > self.n:
            self.recent_wfitted = self.recent_wfitted[1:]
        if len(self.recent_hfitted) > self.n:
            self.recent_hfitted = self.recent_hfitted[1:]
            
        self.bestw = np.mean(self.recent_wfitted)
        self.besth = np.mean(self.recent_hfitted)
        