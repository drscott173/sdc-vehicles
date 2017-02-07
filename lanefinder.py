#
# Udacity Self Driving Car Nanodegree
#
# Project 3
# Advanced Lane Finding
#
# Scott Penberthy
# January, 2016
#

import os, cv2, pickle
import numpy as np
from glob import glob
from moviepy.editor import VideoFileClip
from IPython.display import HTML

#
## Globals 
##
## We keep global variables, hyperparameters and utility functions
## in a singleton of Globals.  We pass this around to all other
## objects.
#

class Globals:

    def __init__(self):
        self.set_hyper_parameters()
        self.load_camera_distortion()
        self.create_kernels()
        self.create_perspective()
        self.make_clip_regions()

    def set_hyper_parameters(self):
        self.img_shape = (360, 640) # working shape
        self.img_size = (self.img_shape[1], self.img_shape[0]) # cv2 reverses this
        self.max_x = self.img_shape[0]-1  # max x for traversing images (vertical)
        self.max_y = self.img_shape[1]-1  # max y for traversing images (horizontal)
        self.n_fits = 10 # number of previous curve fittings to keep
        self.dfit_threshold = (5.0,100.0)  # acceptable range in fit parameter change
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meteres per pixel in x dimension
        self.dx = 30 # scanline skipping in x direction
        self.alpha  = 0.3 # alpha blending for road overlay
        self.n_hist_cutoff = 0.0  # vertical percentage to start histograms
        self.n_inches_per_lane = 12*12 # as it says
        self.n_scale_range = np.array([3.5, 7.0])/2.0  # acceptable scale of pixels/inch

    def load_camera_distortion(self):
        # We assume the distortion matrices are pre-computed and
        # stored in ./calibration/calibration.p as a dictonary
        # with 'mtx' and 'dist' entries.
        self.cbase = "./calibration/"
        info = pickle.load(open(self.cbase+"calibration.p", "rb"))
        self.mtx= info['mtx']
        self.dist = info['dist']

    def create_kernels(self):
        # We create kernels once as we use them for every image
        # First, a kernal to look for white thresholds
        kernel = np.ones((11,11),np.float32)/(1-11*11)
        kernel[5,5] = 1.0
        self.white_2D_kernel = kernel
        #
        # A clahe kernel for image enhancement
        #
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #
        # All coordinates in our image, for creating our lane image
        #
        self.coords = np.indices(self.img_shape) # we could do this just once

    def create_perspective(self):
        # The src points draw a persepective trapezoid, the dst points draw
        # them as a square.  M transforms x,y from trapezoid to square for
        # a birds-eye view.  M_inv does the inverse.
        src = np.float32(((500, 548), (858, 548), (1138, 712), (312, 712)))*0.5
        dst = np.float32(((350, 600), (940, 600), (940, 720), (350, 720)))*0.5
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

    def make_clip_regions(self):
        # We clip the bottom of the birds-eye view to eliminate reflections
        # from the car dashboard.  The roi_clip cuts a trapezoid from a normal
        # image.
        self.bottom_clip = np.int32(np.int32([[[60,0], [1179,0], [1179,650], [60,650]]])*0.5)
        self.roi_clip =  np.int32(np.int32([[[640, 425], [1179,550], [979,719],
                              [299,719], [100, 550], [640, 425]]])*0.5)

    def region_of_interest(self, img, vertices):
        # We preserve the area described by vertices, in clockwise motion,
        # from an input image img and return the result.
        mask = np.zeros_like(img)   
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        cv2.fillConvexPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def smooth(self, x, window_len=32):
        # We apply a 32x32 hanning filter to smooth our noisy 1-D
        # histogram of pixel intensity.  This helps find peaks for lanes.
        w = np.hanning(window_len)
        s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
        return np.convolve(w/w.sum(),s,mode='valid')

    def anneal(self, img1, img2):
        # Average/lerp the values of two images.  I find that when I drive
        # my brain hits steady state and lines blur into a stream of bits
        # on the left and right.  My eye fills in the rest.  We repeat that
        # here by blending the last 10-20 images together.
        return cv2.addWeighted(img1,0.1,img2,0.9,0)

    def peak_window(self, h, idx):
        # Find either sides of a peak that occurs at index idx within
        # a 1-D histogram h.  Return them as [lo,hi] index bounds.
        lo = idx
        hi = idx
        while lo > 0 and h[lo] > 0.05:
            lo += -1
        while hi < len(h) and h[hi] > 0.05:
            hi += 1
        return lo, hi

    def correct_image(self, img):
        # Apply a smoothing filter to dynamically equalize the brightness
        # in an image, pulling out and enhancing true colors.
        # create a CLAHE object (Arguments are optional).
        #
        # See https://en.wikipedia.org/wiki/Adaptive_histogram_equalization
        #
        blue = self.clahe.apply(img[:,:,0])
        green = self.clahe.apply(img[:,:,1])
        red = self.clahe.apply(img[:,:,2])
        img[:,:,0] = blue
        img[:,:,1] = green
        img[:,:,2] = red
        return img

    def bw(self, plane):
        # convert a plane of pixel values (0-255) into a black & white image
        bw = np.zeros((plane.shape[0], plane.shape[1], 3), dtype=np.float32)
        ramped = plane
        bw[:,:,0] = ramped
        bw[:,:,1] = ramped
        bw[:,:,2] = ramped
        return bw

    def fit_y(self, f, x):
        # Given an x value, determine the y out by applying
        # a polynomial fit function f(x).  f is a coefficient vector.
        return f[0]*x**2 + f[1]*x + f[2]

    def dfit(self, f1, f2):
        # Compute the difference between two fit functions f1, f2
        # which we'll use for outlier detection.
        raw = np.abs((f1-f2)/f2)
        return np.mean(raw)

#
## ImageLoader
## A handy class for loading images and retaining their rgb matrix value.
##
## i = ImageLoader(some_path)
## i.rgb
#

class ImageLoader:
    def __init__(self, path):
        raw = cv2.imread(path)
        self.rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
#
## Frame
##
## A frame instance is instantiated with an rgb image of shape (x,y,3).  We
## compute a "mask" instance variable that applies all our tricks to finding
## bits that are candidates for lane edge detection.  We  keep around multiple
## versions of the original image in yuv, hsv, gray and an enhanced "base" 
## version.  This class reduces memory consumption for our data pipeline.
##
#

class Frame:

    def __init__(self, rgb, g):
        # We expect an rgb image as input along with a singleton for Globals, g.
        self.raw = rgb
        self.g = g
        self.normalize()
        self.create_mask()

    def normalize(self):
        # Reduce our image to our chosen processing size.  Remove camera
        # distortion and create base versions of the image in different 
        # color spaces for processing.
        step1 = cv2.resize(self.raw, self.g.img_size, interpolation = cv2.INTER_CUBIC)
        step2 = cv2.undistort(step1, self.g.mtx, self.g.dist, None, self.g.mtx)
        step3 = self.g.correct_image(step2)
        self.base = step3
        self.rgb = cv2.warpPerspective(self.base, self.g.M, self.g.img_size, flags=cv2.INTER_LINEAR)
        self.hsv = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2HSV)
        self.yuv = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2YUV)
        self.gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)

    def create_mask(self):
        # We find candidate pixels by a set of successive filters, adding or
        # removing bits from a final "mask."  We first identify all yellow and
        # white bits, the color of lane markers.  We then use edge detection
        # with a sobel filter.  Finally, we mask off shadows to avoid being
        # confused by trees, clouds, cars.
        self.mask = np.zeros((self.rgb.shape[0], self.rgb.shape[1]), dtype=np.uint8)
        self.add_yellow_bits()
        self.add_white_bits()
        self.add_sobel_bits()
        self.ignore_shadows()
        self.mask = self.g.region_of_interest(self.mask, self.g.bottom_clip)
        return self.mask

    def add_yellow_bits(self):
        # Yellow bits are identified through a band-pass filter in the HSV
        # (hue, saturation, value/brightness) space.
        lower  = np.array([ 0, 80, 200])
        upper = np.array([ 40, 255, 255])
        yellows = np.array(cv2.inRange(self.hsv, lower, upper))
        self.mask[yellows > 0] = 1

    def add_white_bits(self):
        #
        # Method for white filtering as seen in
        # "Real-Time Lane Detection and Rear-End Collision 
        # Warning SystemOn A Mobile Computing Platform", Tang et.al., 2015
        #
        # https://www.researchgate.net/publication/275963307_Real-Time_Lane_Detection_and_Rear-End_Collision_Warning_System_on_a_Mobile_Computing_Platform
        y = self.yuv[:,:,0]
        whites = np.zeros_like(y)
        bits = np.where(y  > 100)  # was 175
        whites[bits] = 1
        mask2 = cv2.filter2D(y,-1,self.g.white_2D_kernel)
        whites[mask2 < 5] = 0
        self.mask = self.mask | whites 

    def add_sobel_bits(self):
        # From class and experimentation, we found that the green plane in
        # an rgb image, and the gray image are good candidates for finding
        # edges.
        green = self.abs_sobel_thresh(self.rgb[:,:,1])
        shadows = self.abs_sobel_thresh(self.gray, thresh_min=10, thresh_max=64)
        self.mask = self.mask | green | shadows

    def abs_sobel_thresh(self, gray, orient='x', thresh_min=20, thresh_max=100):
        # We apply a Sobel filter to find edges, scale the results
        # from 1-255 (0-100%), then use a band-pass filter to create a mask
        # for values in the range [thresh_min, thresh_max].
        sobel = cv2.Sobel(gray, cv2.CV_64F, (orient=='x'), (orient=='y'))
        abs_sobel = np.absolute(sobel)
        max_sobel = max(1,np.max(abs_sobel))
        scaled_sobel = np.uint8(255*abs_sobel/max_sobel)
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        return binary_output

    def ignore_shadows(self):
        # Shadows are an issue when detecting lanes, as the shadow edges
        # create false positives.  I follow a simple intuition -- look for the
        # brighter spots on the road and ignore the really dark areas.
        bits = np.zeros_like(self.gray)
        thresh = np.mean(self.gray)
        bits[self.gray > thresh] = 1
        self.mask = self.mask & bits

#
## Line
## A "line" represents the left or right lane marker on the road.  We
## feed instances successive polynomials that were fit to pixel data
## taken from the road image.  We call these polynomials a "fit."  The
## lane class examines these, determines if they're junk, and keeps the
## good ones around.  The class computes a moving average of the last N
## fits as the lane polynomial, also storing the curvature computed closest
## to the car at the bottom of the image frame.
##
## l = Line(g), where g is the Global singleton
## l.update_history(fit), repeatedly
#


class Line:
    def __init__(self, g):
        self.g = g
        self.reset()

    def reset(self):
        self.fits = []
        self.fit = None
        self.curvature = None
        self.curvatures = []
        self.marker = None
        self.weak_counter = 0
        self.pixels = None

    def update_history(self, fit, weak=False):
        # We introduce another polynomial that was "fit" to pixel data
        # as a candidate for this line.  We pass in a boolean, weak,
        # to indicate if we've used a pre-existing "fit" when pixels
        # failed to work.
        # 
        # We track failed fits with a weak_counter to detect when
        # its time to try more thorough approaches.
        #
        # We set our current fit, self.fit, to the moving average
        # of the last N fits.  We compute our curvature, too.  We then
        # add both the fit and curvature to our history.
        if weak or (fit is None):
            self.weak_counter += 1
            return
        self.fit = self.moving_average(fit)
        c = self.get_curvature()
        if c != np.inf and c != -np.inf:
            self.curvatures.append(c)
        self.curvature = c

    def line_points(self):
        # Return a set of points in image space that represent
        # our fit function.  This is useful for extracting pixels
        # form an existing image when looking for the "next" location
        # of the lane and for adjusting initial guesses based on a small
        # window. 
        pts = []
        for x in range(0, self.g.max_x, self.g.dx):
            y = int(0.5+self.g.fit_y(self.fit, x))
            if y >= 0 and y < self.g.max_y:
                pts.append([x, y])
        return np.array(pts)

    def get_curvature(self):
        # Compute the curvature of our current fit, which is the radius
        # of a circle that draws the arc upon which we drive.  These are often
        # measured in thousands of meters.   We ballpark it by transforming
        # from perspective space to euclidean space with constants set in
        # Globals.
        if self.fit is None:
            return np.inf
        y_eval = self.g.max_x
        pts = self.line_points()
        fit_cr = np.polyfit(pts[:,0]*self.g.ym_per_pix, pts[:,1]*self.g.xm_per_pix, 2)
        curverad = ((1 + (2*fit_cr[0]*y_eval + fit_cr[1])**2)**1.5) \
                                     /np.absolute(2*fit_cr[0])
        return curverad

    def moving_average(self, fit):
        # Simple moving average calculation, ensuring we only use n_fits
        # samples.
        while len(self.fits) > self.g.n_fits:
            self.fits = self.fits[1:]
        if fit is not None:
            if len(self.fits) > 0:
                df = self.g.dfit(fit, self.fits[-1])
                if df > self.g.dfit_threshold[0] and df < self.g.dfit_threshold[1]:
                    #print("dfit ", df)
                    return self.fits[-1]
            self.fits.append(fit) 
        if len(self.fits):
            return np.mean(np.array(self.fits), axis=0)
        return None

    def pixel_fit(self, frame, window=60):
        # Use our current fit to project a path on the current frame,
        # then pull pixels within +/- the window of our curve.  Fit a new
        # curve to the pixels we draw.  Blank regions or bad fits
        # set "weak" to True.  Weak implies we have to guess the fit
        # as our predictive methods have failed.
        self.pixels = np.zeros_like(frame.mask)
        if self.fit is None:
            return self.pixels
        for x in range(self.g.max_x, 0, -1):
            y = int(0.5+self.g.fit_y(self.fit, x))
            lo_y = max(0, y-window)
            hi_y = min(self.g.max_y, y+window)
            #print("x,y_lo,y_hi = ", x, lo_y, hi_y)
            self.pixels[x,lo_y:hi_y] = frame.mask[x,lo_y:hi_y]
        x,y = np.where(self.pixels > 0)
        try:
            new_fit = np.polyfit(x, y, 2)
            weak = False
        except:
            weak = True
            print("Weak!")
            new_fit = fit
        self.update_history(new_fit, weak)
#
## LaneBoundaries
## A simple data structure for holding the starting (lo) and 
## ending (hi) pixels for the left (l) and right (r) lanes.
#

class LaneBoundaries:
    def __init__(self):
        self.l_lo = 0
        self.l_hi = 0
        self.r_lo = 0
        self.r_hi = 0

    def all(self):
        return self.l_lo, self.l_hi, self.r_lo, self.r_hi


#
## Lane
## A lane consists of a left and right line.   We feed the lane images
## and it determines the left and right lines of our driving scene.
##
## lane = Lane(g), where g is a singleton of Globals
## lane.process_image(rgb), an rgb image captured from our front-facing camera
##
#

class Lane:
    def __init__(self, g):
        self.g = g
        self.reset()

    def reset(self):
        # Reset our metadata 
        self.lane_pixels = None    # the width of our lane in pixels
        self.scale = None  # assuming we're driving in the US, the pixel scale
        self.pixels = None # the mask pixels we used for fitting lanes
        self.left = Line(self.g) # our left lane marker
        self.right = Line(self.g) # our right lane marker
        self.frame = None # the current frame, created from an image
        self.frame_counter = 0  # the number of frames since reset

    def process_image(self, image):
        #
        # Ingest an image.  If we have markers, use the existing fit curves
        # to find pixels.  If not, do the more complex scan of bands of an image
        # finding areas of highest intensity (a histogram fit), then extracting
        # candidate curves, followed by the regular pixel fit.
        #
        self.frame = Frame(image, self.g)
        self.frame_counter += 1
        if not self.we_have_markers():
            self.histogram_fit()
        self.pixel_fit()
        self.update_stats()
        return self.visualize()

    def we_have_markers(self):
        # Return True if we have two live markers
        return (self.left.fit is not None) and (self.right.fit is not None)

    def histogram_fit(self,cutoff=0):
        #
        # Use intensity histograms in the x (vertical) direction to detect
        # potential lane markers on the left and ride side of the image.
        # Feed these to our makers who will figure out if they're good and
        # compute a moving average, which we'll use for guidance.
        #
        edges = self.frame.mask
        h = self.g.smooth(np.mean(edges[np.int(edges.shape[0]*self.g.n_hist_cutoff):,:], axis=0))
        b = self.lane_boundaries(h)
        self.pixels, left_fit, right_fit = self.fit_line_to_frame(b)
        self.left.update_history(left_fit)
        self.right.update_history(right_fit)

    def pixel_fit(self):
        #
        # Use existing markers to trace a path on our image, extracting
        # pixels within a window, then fitting a curve to the result.
        # Combine the pixels from left and right analysis into the lane
        # analysis for debugging.
        #
        self.left.pixel_fit(self.frame)
        self.right.pixel_fit(self.frame)
        self.pixels = self.left.pixels | self.right.pixels

    def update_stats(self):
        # 
        # After every frame compute our lane markers, curvature, and position
        # in the frame.  Make sure everything is OK and update our fit curves
        # as needed.
        #
        self.reset_stats()
        self.compute_lane_markers()
        self.measure_lane()
        self.check_if_ok()

    def reset_stats(self):
        # Reset our lane metadata prior to testing everything.
        self.left_marker = None
        self.right_marker = None
        self.scale = None
        self.off_center = None
        self.lane_pixels = None

    def compute_lane_markers(self):
        # Compute our initial position of the lane markers by evaluating
        # our fit curves at the bottom of the image.
        x = self.g.max_x 
        if self.left.fit is not None: 
            self.left_marker = self.g.fit_y(self.left.fit, x) 
        if self.right.fit is not None:
            self.right_marker = self.g.fit_y(self.right.fit, x)

    def measure_lane(self):
        # Assume that a lane of a given width must be centered in the frame
        # to mean we're centered on the physical road.  We'd want to calibrate
        # any offset bias in the real world.  For now we track from the center
        # and report deviation in in inches.
        if self.left_marker is not None and self.right_marker is not None:
            self.lane_pixels = max(2,self.right_marker - self.left_marker)
            centered_left = (self.g.max_y - self.lane_pixels)*0.5
            self.off_center = (centered_left - self.left_marker)/self.lane_pixels
            self.off_center = self.g.n_inches_per_lane*self.off_center
            self.scale = self.lane_pixels / self.g.n_inches_per_lane

    def check_if_ok(self):
        # We have a good lane if the curves exist, and if the pixels that separate
        # them are a reasonable road width.  We capture this in "scale" which means
        # the number of pixels per inch we're seeing assuming a road in 12 feet wide.
        self.ok = (self.left.fit is not None)
        self.ok = self.ok and (self.right.fit is not None)
        self.ok = self.ok and self.scale_ok()
        return self.ok

    def lane_boundaries(self, h):
        # Given an intensity histogram h, find the peaks on left an right,
        # then the window of intensity on either side of the peak that exceeds
        # a noise threshold.
        b = LaneBoundaries()
        midpoint = int(len(h)/2)
        b.l_lo, b.l_hi = self.g.peak_window(h, np.argmax(h[0:midpoint]))
        b.r_lo, b.r_hi = self.g.peak_window(h, midpoint+np.argmax(h[midpoint:]))
        return b

    def copy_lane_bits(self, src, dst, boundaries):
        # Utility function to copy bits from a src image to a destination
        # image, taking full vertical rectangles bounded by y in [l_lo, l_hi]
        # on the left, and y in [r_lo, r_hi] on the right.
        l_lo, l_hi, r_lo, r_hi = boundaries.all()
        dst[:,l_lo:l_hi] = src[:,l_lo:l_hi]
        dst[:,r_lo:r_hi] = src[:,r_lo:r_hi]

    def fit_line_to_frame(self, boundaries):
        #
        # Given [min, max] boundaries for our left and right lanes,
        # use the pixels in the mask to fit a curve for the left and
        # right.  If we fail due to lack of pixels, we set Weak to True.
        #
        # Return a clean mask of pixels used for the fit, then the
        # left and right quadratic coefficients a,b,c.
        clean = np.zeros_like(self.frame.mask)
        mid = int(clean.shape[1]/2)
        self.copy_lane_bits(self.frame.mask, clean, boundaries)
        x_l, y_l = np.where(clean[:,0:mid] == 1)
        x_r, y_r = np.where(clean[:,mid:] == 1)
        try:
            fit_fn_l = np.polyfit(x_l, y_l, 2)
        except:
            print("Left fit fail")
            fit_fn_l = None
        try:
            fit_fn_r = np.polyfit(x_r, y_r+mid, 2)
        except:
            print("Right fit fail")
            fit_fn_r = None
        return clean, fit_fn_l, fit_fn_r 

    def scale_ok(self):
        # Assume the lane markers represent a road in the US.  Assume the
        # road is 12 feet wide.  How many pixels per inch is this?  We compute
        # this as the "scale" and then make sure its within an acceptable range.
        ok = (self.scale is not None)
        ok = ok and self.scale >= self.g.n_scale_range[0]
        ok = ok and self.scale <= self.g.n_scale_range[1]
        return ok

    def visualize(self):
        #
        # Create a visualization of our data pipeline and the final
        # lane markers drawn on the source image.  We use this for
        # debugging and analysis.
        # 
        self.create_diagnostics()
        diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        diagScreen[0:720, 0:1280] = self.large_diag(6)
        diagScreen[720:840, 0:1280] = self.create_status_panel()
        for i in range(0,6):
            diagScreen[840:1080, i*320:(i+1)*320] = self.pipeline_diag(i) 
        for i in range(0,3):
            diagScreen[240*i:(i+1)*240, 1280:1600] = self.sobel_diag(i)
        hls = cv2.cvtColor(self.frame.base, cv2.COLOR_RGB2HLS)
        for i in range(0,3):
            diagScreen[240*i:(i+1)*240, 1600:1920]= self.hls_diag(hls, i)
        self.mosaic = cv2.resize(diagScreen, (1280,720), interpolation=cv2.INTER_AREA)
        return self.mosaic

    def histogram_chart(self, edges):
        #
        # Draw a histogram of intensity over a bitmap of candidate
        # edge pixels in edges.  Return this image for diagnostics.
        #
        histogram = np.mean(edges[np.int(edges.shape[0]*self.g.n_hist_cutoff):,:], axis=0)
        clean = self.g.smooth(histogram)
        chart = self.g.bw(edges*255)
        max_x = int(chart.shape[0]*0.9)
        x_scalar = float(max_x*0.8/max(1,np.max(clean)))
        y_scalar = float(chart.shape[1]-1)/len(clean)
        pts = []
        for i in range(0,len(clean)):
            ix = max_x-int(x_scalar*clean[i])
            iy = int(y_scalar*i)
            pts.append((iy,ix))
        pts = sorted(pts)
        for i in range(1,len(pts)):
            cv2.line(chart, pts[i-1], pts[i], (255, 0, 0), 5)
        return chart

    def draw(self):
        #
        # Draw our current lane on a black and white image, where 1's
        # are the lane and 0's are the environment.
        #
        lane = np.zeros_like(self.frame.mask)
        coords = self.g.coords
        y = coords[0]
        #lane_stats = wheres_my_lane(warped_size, left_fit, right_fit)
        if (self.left.fit is not None) and (self.right.fit is not None):
            lane[coords[1] >= self.g.fit_y(self.left.fit, y)] = 1
            lane[coords[1] > self.g.fit_y(self.right.fit, y)] = 0
        self.lane_image = lane
        return lane

    def draw_overlay(self):
        #
        # Take our lane image (in birds eye view), warp it back to
        # a perspective view, and overlay lay this on our source image.
        # Store as self.overlay
        #
        warped_back = cv2.warpPerspective(self.lane_image, self.g.M_inv, 
            self.g.img_size, flags=cv2.INTER_LINEAR)
        road = np.zeros_like(self.frame.base)
        road[:,:,1] = warped_back*255
        self.overlay = np.zeros_like(self.frame.base)
        cv2.addWeighted(road, self.g.alpha, self.frame.base, 1 - self.g.alpha, 0, self.overlay)
        return self.overlay

    def create_diagnostics(self):
        # Collect diagnostic images for debugging and display.
        diags = [self.frame.raw]
        diags.append(self.frame.base)
        diags.append(self.frame.rgb)
        diags.append(self.histogram_chart(self.frame.mask))
        diags.append(self.g.bw(self.pixels*255))
        diags.append(self.g.bw(self.draw()*255))
        diags.append(self.draw_overlay())
        self.diags = diags 
        return diags

    def pipeline_diag(self, n):
        # Create a cell for the visualization showing an element of our
        # data pipeline, which are stored in our diags array.
        return cv2.resize(self.diags[n], (320,240), interpolation=cv2.INTER_AREA)

    def large_diag(self, n):
        # Create a large cell for the visualization showing an element of our
        # pipeline, which are stored in our diags array.
        return cv2.resize(self.diags[n], (1280, 720), interpolation=cv2.INTER_AREA)

    def hls_diag(self, hls, n):
        # Create an HLS image from the nth plane (h, l, s) and return
        # an image for visualization.
        return cv2.resize(self.g.bw(hls[:,:,n]), (320, 240), interpolation=cv2.INTER_AREA)

    def sobel_diag(self, n):
        # Create a diagnostic image showing a sobel filter applied to
        # an rgb plane n of our original image, where 0=red, 1=green, 2=blue.
        plane = self.frame.raw[:,:,n]
        sobelized = self.frame.abs_sobel_thresh(plane)
        rgb = np.zeros((plane.shape[0], plane.shape[1], 3), dtype=np.float32)
        rgb[:,:,n] = sobelized*255
        return cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA)

    def create_status_panel(self):
        # Create an image with text info describing our current frame
        # and its statistics.
        font = cv2.FONT_HERSHEY_COMPLEX
        middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)
        s = self.scale
        if s is None:
            s = 0
        lc = self.left.curvature
        rc = self.right.curvature
        scale_str = 'Frame {} scale {:.4f} curve {:.0f},{:.0f}'
        scale_str = scale_str.format(self.frame_counter,s,lc,rc)
        cv2.putText(middlepanel, scale_str, (30, 60), font, 1, (255,0,0), 2)
        offc = self.off_center
        if offc is None:
            offc = 0
        off_str = 'Inches to right of center: {:.4f}'.format(offc)
        cv2.putText(middlepanel, off_str, (30, 90), font, 1, (255,0,0), 2)
        return middlepanel

#
## Testing
##
## Here we diverge from our object classes to use procedural APIs
## to test whether our logic is working on vidoes and images of
## road lanes.
#

g = Globals()  # a singleton of globals
lane = Lane(g) # a smart "lane" we feed images to find the edges

def process_image(image):
    #
    # Process a single image of driving lanes and store the output
    # in frameN.jpg, where N is the current frame counter of our global
    # lane test object.
    #
    global g, lane
    if lane.frame is not None:
        image = g.anneal(image, lane.frame.raw)
    output = lane.process_image(image)
    write_name = "./movies/frame{}.jpg".format(lane.frame_counter)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(write_name, bgr)
    if not lane.ok:
        err=':('
        if lane.left.fit is None:
            err += ' No left'
        if lane.right.fit is None:
            err += ' No right'
        if not lane.scale_ok():
            err += ' Bad size {}'.format(lane.scale)
        print(err)
    return output

def process_video(input_path, output_path):
    #
    # Process a video at input_path and store the result
    # in output_path.
    #
    global g, lane
    clip1 = VideoFileClip(input_path)
    lane.reset()
    test_clip = clip1.fl_image(process_image)  
    test_clip.write_videofile(output_path, audio=False)

def do_project():
    #
    # Run our code on the assigned project video.
    #
    test_input = '../CarND-Advanced-Lane-Lines/project_video.mp4'
    test_output = './movies/new_project.mp4'
    process_video(test_input, test_output)

def do_challenge():
    #
    # Run our code on a challenge video. 
    #
    test_input = '../CarND-Advanced-Lane-Lines/challenge_video.mp4'
    test_output = './movies/new_challenge.mp4'
    process_video(test_input, test_output)   

def test_it():
    # Here's a sanity testing function to make sure our routines are
    # basically working.  We grab a few images and try to process them.
    g = Globals()
    i1 = ImageLoader("./movies/frame14.jpg")
    i2 = ImageLoader("./movies/frame15.jpg")
    i3 = ImageLoader("./movies/frame16.jpg")
    i4 = ImageLoader("../CarND-Advanced-Lane-Lines/test_images/test1.jpg")
    robot = Lane(g)
    out1 = robot.process_image(i1.rgb)
    out2 = robot.process_image(i2.rgb)
    out3 = robot.process_image(i3.rgb)
    return [robot, out1, out2, out3]



