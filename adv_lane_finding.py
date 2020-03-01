import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
# %matplotlib qt

class Camera:
    def __init__(self, dir_input, dir_output):
        self.mtx, self.dist, self.rvecs, self.tvecs = self.calibrateCamera(dir_input, dir_output)
        self.M, self.M_inv = self.calc_perspective_trnsfm_matrix()
    
    def calc_perspective_trnsfm_matrix(self, margin=300):
        src = np.float32([[(595,450),(690,450),(1108,719),(210, 719)]])
        dst = np.float32([(margin,0),(1280-margin,0),(1280-margin,719),(margin,719)])
        M = cv2.getPerspectiveTransform(src, dst)
        M_inv = cv2.getPerspectiveTransform(dst, src)
        return M, M_inv

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

from moviepy.editor import VideoFileClip
from IPython.display import HTML

def find_filename(path_to_file):
    return path_to_file.split('/')[-1]

class LaneFinding:
    def __init__(self, camera):
        """camera a Camera object"""
        self.camera = camera
        pass

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
    
    def find_lane_pixels(self, binary_warped):
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
        minpix = 50

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

        return leftx, lefty, rightx, righty, out_img
    
    def fit_polynomial(self, binary_warped):
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(binary_warped)

        # Fit a second order polynomial to each using `np.polyfit`
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        
        left_fitx = (left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2])
        right_fitx = (right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2])

        ## Visualization ##
        # Colors in the left and right lane regions
        # out_img[lefty, leftx] = [255, 0, 0]
        # out_img[righty, rightx] = [0, 0, 255]

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
        cv2.fillPoly(out_img, np.int_([pts]), (0,255, 0))

        return out_img

    def process_image(self, img):
        """return a color image"""
        undist_img = self.camera.undistort_image(img)
        bin_dir = self.sobel_dir_select(undist_img, 15, (0.7,1.3))
        bin_mag = self.sobel_mag_select(undist_img,15,(100,255))
        bin_s = self.hls_select(undist_img, thresh=(120,255))
        bin_sobelx = self.sobel_1Dselect(undist_img,'x',15,(30,255))
        bin_sobely = self.sobel_1Dselect(undist_img,'y',15,(30,255))
        # Combine filters
        bin_img = (bin_s | (bin_sobelx&bin_sobely & bin_dir))*255
        bin_warped_img = self.camera.perspective_trnsfm(bin_img, self.camera.M)
        lane_mark_mask = self.fit_polynomial(bin_warped_img)
        lane_mark_mask = self.camera.perspective_trnsfm(lane_mark_mask, self.camera.M_inv)
        result = cv2.addWeighted(undist_img, 1, lane_mark_mask, 0.3, 0)
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

myLaneFinding = LaneFinding(Camera('camera_cal','output_cal'))
myLaneFinding.process_video('project_video.mp4')

# # Create folder to hold undistort images
# dir_output_undist = 'output_undist'
# if not os.path.exists(dir_output_undist):
#     os.mkdir(dir_output_undist)
# # Generate undistort images to verify
# for fname in images:
#     img = cv2.imread(fname)
#     dst = cv2.undistort(img, mtx, dist, None, None)
#     cv2.imwrite(os.path.join(dir_output_undist,fname.split('/')[-1]), dst)

# img = cv2.imread('./test_images/straight_lines1.jpg')
# dst = cv2.undistort(img, mtx, dist, None, None)
# cv2.imwrite('test.jpg',dst)



# # Define source and destination points for perspective transform