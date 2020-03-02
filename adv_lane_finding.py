import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
# %matplotlib qt

class Camera:
    def __init__(self, dir_input, dir_output):
        self.mtx, self.dist, self.rvecs, self.tvecs = self.calibrateCamera(dir_input, dir_output)
        self.M, self.M_inv, self.xm_per_pix, self.ym_per_pix = self.calc_perspective_trnsfm_matrix()
    
    def calc_perspective_trnsfm_matrix(self, margin=300):
        ym_per_pix = 3/100 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        src = np.float32([[(577,465),(706,465),(1108,719),(210, 719)]])
        dst = np.float32([(margin,0),(1280-margin,0),(1280-margin,719),(margin,719)])
        M = cv2.getPerspectiveTransform(src, dst)
        M_inv = cv2.getPerspectiveTransform(dst, src)
        return M, M_inv, xm_per_pix, ym_per_pix

    def perspective_trnsfm(self, img, matrix):
        warped = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        return warped

    def undistort_image(self, img):
        """img an object return by cv2.imread"""
        return cv2.undistort(img, self.mtx, self.dist, None, None)
    def calibrateCamera(self,input_dir, output_dir):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Image Size
        shape_y = 720
        shape_x = 1280

        # Make a list of calibration images
        images = glob.glob(input_dir+'/calibration*.jpg')

        # Create folder to save calibration outputs
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            cv2.resize(img, (shape_y, shape_x))
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                cv2.imwrite(os.path.join(output_dir,fname.split('/')[-1]),img)

        # Camera Calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (shape_y, shape_x), None, None)

        return mtx, dist, rvecs, tvecs

# myCamera = Camera()
# Define a class to receive the characteristics of each line detection
class Line():

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature for the most recent fit
        self.current_curvature = None
        #radius of curvature of the line in some units
        self.best_curvature = None 
        #distance in meters of vehicle center from the line
        self.current_line_base_pos = None 
        self.best_line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

    def eval_best_fit(self, ploty):
        return self.best_fit[0]*ploty**2 + self.best_fit[1]*ploty + self.best_fit[2]
    
    def eval_current_fit(self, ploty):
        return self.current_fit[0]*ploty**2 + self.current_fit[1]*ploty + self.current_fit[2]

    def update_current_fit(self, allx, ally):
        self.current_fit = np.polyfit(ally, allx, 2)
        if self.best_fit is None:
            self.best_fit = self.current_fit
        self.allx = allx
        self.ally = ally
        self.current_line_base_pos = self.eval_current_fit(719)
        if self.best_line_base_pos is None:
            self.best_line_base_pos = self.current_line_base_pos

    def update_current_curvature(self, y_pix, camera):
        """calculate R_curve (radius of curvature) based on the current fit"""
        y = y_pix/camera.ym_per_pix
        fit_real = [camera.xm_per_pix*self.current_fit[0]/(camera.ym_per_pix**2), camera.xm_per_pix*self.current_fit[1]/camera.ym_per_pix, self.current_fit[2]*camera.xm_per_pix]
        self.current_curvature = ((1 + (2*fit_real[0]*y*camera.ym_per_pix + fit_real[1])**2)**1.5) / np.absolute(2*fit_real[0])
        if self.best_curvature is None:
            self.best_curvature = self.current_curvature

    def update_best_line_base_pos(self, filter_coeff = 0.9):
        if self.best_line_base_pos is None:
            self.best_line_base_pos = self.current_line_base_pos
        else:
            self.best_line_base_pos = self.current_line_base_pos*filter_coeff + (1-filter_coeff)*self.best_line_base_pos
    
    def update_best_fit(self, filter_coeff = 0.9):
        if self.best_fit is None:
            self.best_fit = self.current_fit
        else:
            self.best_fit = self.current_fit*filter_coeff + (1 - filter_coeff)*self.best_fit

    def update_best_curvature(self, filter_coeff = 0.9):
        """
        Calculates the curvature of polynomial functions in meters.
        """
        if self.best_curvature is None:
            self.best_curvature = self.current_curvature
        else:
            self.best_curvature = self.current_curvature*filter_coeff + (1 - filter_coeff)*self.best_curvature

    def self_check(self):
        if self.best_curvature is None or self.current_curvature is None or self.best_line_base_pos is None:
            return True
        elif (np.absolute(self.best_curvature - self.current_curvature) <= 300) and np.absolute(self.best_line_base_pos - self.current_line_base_pos) <= 100:
            return True
        else:
            return False

    def check_parallel(self, other, camera):
        y_max = 719
        y_min = 0
        # Calculate distance at the top
        left_fitx = self.current_fit[0]*y_min**2 + self.current_fit[1]*y_min + self.current_fit[2]
        right_fitx = other.current_fit[0]*y_min**2 + other.current_fit[1]*y_min + other.current_fit[2]
        dist_top = right_fitx - left_fitx
        # Calculate distance at the bottom
        left_fitx = self.current_fit[0]*y_max**2 + self.current_fit[1]*y_max + self.current_fit[2]
        right_fitx = other.current_fit[0]*y_max**2 + other.current_fit[1]*y_max + other.current_fit[2]
        dist_bot = right_fitx - left_fitx
        # Metric
        dist_avg = np.mean([dist_bot, dist_top])*camera.xm_per_pix
        dist_diff = np.absolute((dist_top - dist_bot)*camera.xm_per_pix)

        if 3.5 <= dist_avg <= 4 and dist_diff <= 0.5:
            return True
        else:
            return False

from moviepy.editor import VideoFileClip
from IPython.display import HTML

def find_filename(path_to_file):
    return path_to_file.split('/')[-1]

class LaneFinding:
    def __init__(self, camera, left_lane, right_lane):
        """camera a Camera object"""
        self.camera = camera
        self.left_lane = left_lane
        self.right_lane = right_lane

    def hls_select(self, img, thresh=(0,255)):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        s = hls[:,:,2]
        bin_img = np.zeros_like(s)
        bin_img[(thresh[0] < s) & (s <= thresh[1])] = 1
        return bin_img

    def sobel_1Dselect(self, img, axis='x', sobel_kernel=3, thresh=(0,255)):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if axis == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        sobel = np.absolute(sobel)/np.max(sobel)*255
        bin_img = np.zeros_like(gray)
        bin_img[(sobel > thresh[0]) & (sobel <= thresh[1])] = 1
        return bin_img

    def sobel_mag_select(self, img, sobel_kernel=3, thresh=(0,255)):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobel = np.sqrt(sobelx**2+sobely**2)
        abs_sobel = (abs_sobel/np.max(abs_sobel)*255).astype(np.uint8)
        bin_img = np.zeros_like(gray)
        bin_img[(abs_sobel > thresh[0]) & (abs_sobel <= thresh[1])] = 1
        return bin_img

    def sobel_dir_select(self, img, sobel_kernel=3, thresh=(0,np.pi/2)):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        sobel_angle = np.arctan2(np.absolute(sobely),np.absolute(sobelx))
        bin_img = np.zeros_like(gray)
        bin_img[(sobel_angle > thresh[0]) & (sobel_angle <= thresh[1])] = 1
        return bin_img

    def search_around_poly(self, binary_warped, margin = 100):
        """Search around last fitted line and return detected pixels"""

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Grab the current ploy fit coefficient
        left_fit = self.left_lane.best_fit
        right_fit = self.right_lane.best_fit
        
        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                        left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                        right_fit[1]*nonzeroy + right_fit[2] + margin)))
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty
    
    def search_in_win(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 500

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Draw the windows on the visualization image
            # cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            # (win_xleft_high,win_y_high),(0,255,0), 2) 
            # cv2.rectangle(out_img,(win_xright_low,win_y_low),
            # (win_xright_high,win_y_high),(0,255,0), 2) 
            
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty
    
    def fit_polynomial(self, binary_warped):
        if self.left_lane.detected and self.right_lane.detected:
            leftx, lefty, rightx, righty = self.search_around_poly(binary_warped)    
        else:
            leftx, lefty, rightx, righty = self.search_in_win(binary_warped)

        # update left and right lane polyfit and update measured curvature
        self.left_lane.update_current_fit(leftx, lefty)
        self.left_lane.update_current_curvature(binary_warped.shape[0]-10, self.camera)
        if self.left_lane.self_check():
            self.left_lane.update_best_curvature(0.9)
            self.left_lane.detected = True
        else:
            self.left_lane.detected = False

        self.right_lane.update_current_fit(rightx, righty)
        self.right_lane.update_current_curvature(binary_warped.shape[0]-10, self.camera)
        if self.right_lane.self_check():
            self.right_lane.update_best_curvature(0.9)
            self.right_lane.detected = True
        else:
            self.right_lane.detected = False
        
        if Line.check_parallel(self.left_lane, self.right_lane, self.camera):
            self.left_lane.detected = True
            self.right_lane.detected = True
            self.left_lane.update_best_fit(0.72)
            self.right_lane.update_best_fit(0.72)
            self.left_lane.update_best_line_base_pos()
            self.right_lane.update_best_line_base_pos()
            # self.left_lane.update_best_curvature(0.72)
            # self.right_lane.update_best_curvature(0.72)
        elif self.left_lane.detected and len(self.left_lane.allx) >= len(self.right_lane.allx):
                # trust left lane detection
                self.left_lane.update_best_fit(0.72)
                self.left_lane.update_best_line_base_pos()
                self.right_lane.current_curvature = self.left_lane.best_curvature
                self.right_lane.update_best_curvature(0.72)
        elif self.right_lane.detected and len(self.left_lane.allx) <= len(self.right_lane.allx):
            # trust right lane detection
            self.right_lane.update_best_fit(0.72)
            self.right_lane.update_best_line_base_pos()
            self.left_lane.current_curvature = self.right_lane.best_curvature
            self.left_lane.update_best_curvature(0.72)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )    
        left_fitx = self.left_lane.eval_best_fit(ploty)
        right_fitx = self.right_lane.eval_best_fit(ploty)

        # Measure the curvature in front of vehicle
        # print(left_curverad, right_curverad)
        

        ## Visualization ##

        # Plots the left and right polynomials on the lane lines
        # cv2.polylines(out_img, np.vstack((left_fitx, ploty)).T.astype("uint32"), False, color=[255, 0, 255])
        # cv2.polylines(out_img, np.array([np.vstack((left_fitx, ploty)).T.astype(np.int32)]), False, color=[255, 0, 255], thickness=10)
        # cv2.polylines(out_img, np.array([np.vstack((right_fitx, ploty)).T.astype(np.int32)]), False, color=[255, 0, 255], thickness=10)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        out_img = np.dstack((binary_warped,binary_warped,binary_warped))
        cv2.fillPoly(out_img, np.int_([pts]), (0,255, 0))
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        return out_img


    def process_image(self, img):
        """return a color image"""
        undist_img = self.camera.undistort_image(img)
        bin_dir = self.sobel_dir_select(undist_img, 15, (0.8,1.2))
        bin_mag = self.sobel_mag_select(undist_img,15,(50,255))
        bin_s = self.hls_select(undist_img, thresh=(120,255))
        bin_sobelx = self.sobel_1Dselect(undist_img,'x',15,(30,255))
        bin_sobely = self.sobel_1Dselect(undist_img,'y',15,(30,255))
        # Combine filters
        # bin_img = (bin_s | (bin_sobelx&bin_sobely & bin_dir))*255
        bin_img = (bin_sobely&bin_sobelx) | (bin_s&bin_dir)
        bin_warped_img = self.camera.perspective_trnsfm(bin_img, self.camera.M)
        lane_mark_mask = self.fit_polynomial(bin_warped_img)
        lane_mark_mask = self.camera.perspective_trnsfm(lane_mark_mask, self.camera.M_inv)
        result = cv2.addWeighted(undist_img, 1, lane_mark_mask, 0.9, 0)
        # Draw text
        text = "Curvature: {:.0f} (m)".format((self.left_lane.best_curvature + self.left_lane.best_curvature)/2)
        self.draw_text(result, text, (20, 100))
        text = "Distance from lane center: {:.1f} (m)".format(((self.left_lane.best_line_base_pos+self.right_lane.best_line_base_pos)/2 - 960)*self.camera.xm_per_pix)
        self.draw_text(result, text, (20, 200))
        return result

    def process_video(self, path_to_video, output_dir = 'output_video', sub_clip = None):
        """clip a VideoFileClip object"""
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if sub_clip is None:
            clip = VideoFileClip(path_to_video)
        else:
            clip = VideoFileClip(path_to_video).subclip(sub_clip[0],sub_clip[1])
        white_clip = clip.fl_image(self.process_image) #NOTE: this function expects color images!!
        white_clip.write_videofile(os.path.join(output_dir, find_filename(path_to_video)), audio=False)

    def draw_text(self, img, text, loc):
        cv2.putText(img, text, loc, cv2.FONT_HERSHEY_SIMPLEX , 2, color=[0,0,0], thickness=2)

def test():
    left_lane = Line()
    right_lane = Line()
    myLaneFinding = LaneFinding(Camera('camera_cal','output_cal'), left_lane, right_lane)
    myLaneFinding.process_video('project_video.mp4',sub_clip=(38,43))

test()