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
        warped_img = self.camera.perspective_trnsfm(bin_img, self.camera.M)
        filtered_img = np.dstack((warped_img,warped_img,warped_img))
        return filtered_img

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
myLaneFinding.process_video('project_video.mp4', sub_clip=(0,5))

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