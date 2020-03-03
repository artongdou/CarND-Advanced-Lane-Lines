## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_imgs/undistort_image.jpg "Undistorted"
[image2]: ./writeup_imgs/undistort_test_image.jpg "Road Transformed"
[image3]: ./writeup_imgs/binary_image.jpg "Binary Example"
[image4]: ./writeup_imgs/perspective.jpg "Warp Example"
[image5]: ./writeup_imgs/fit_poly.png "Fit Visual"
[image6]: ./writeup_imgs/frame17.jpg "Output"
[video1]: ./output_video/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this setp is contained in the class `Camera` in file `adv_lane_finding.py`. The actual steps to calibrate the camera will be the class method `Camera.calibrateCamera`.

For each image, my first step is to resize to (1280x720) using `cv2.resize()`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #522 through #526 in `adv_lane_finding.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is implemented in `Camera.calc_perspective_trnsfm_matrix()` in `adv_lane_finding.py`. This function takes as inputs margin (`margin`) calculate the destination points as follow. And the source points are hardcoded. With the `margin` of 300, it resulted in the following source and destination points:
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 577, 485      | 300, 0        | 
| 706, 465      | 980, 0        |
| 1108, 719     | 980, 719      |
| 210, 719      | 300, 719      |

I verified that my perspective transform was working as expected by transforming a straightline image to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

`LaneFinding.find_lane_mark()` in `adv_lane_finding.py` is the function to detect the lane mark pixels and perform polynomial fit on those pixels to find the ROI bounded by the lanemark. It utilizes 2 different functions `LaneFinding.search_around_poly()` and `LaneFinding.search_in_win()` to find lane pixels. `LaneFinding.search_around_poly()` is only used when both lanes are detected in the previous image. Otherwise, `LaneFinding.search_in_win()` will perform sliding window search based on the histogram filter of the binary image towards the bottom (line #374).

`Line.update_current_fit()` is implemented to perform 2nd order polynomial fit on detected pixels like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

`Line.update_current_curvature()` is implemented to calculate the curvature of the current polynomial fitted line near the bottom of the image. `Line.update_best_curvature()` is implemented to apply a first order filter to filter out the noise from the calculated curvature frame by frame. The pipeline only call `Line.update_best_curvature()` when there is good detection on the lane (line #452 to #470).

In case of one lane is detected and the other is not, the pipeline uses the curvature of the detected lane to feed in `Line.update_best_curvature()` (line #473 to #493).

To calculate the distance with respect to center, `Line.best_line_base_pos` is used to keep track of the latest filtered base position of the lane. By averaging the `Line.best_line_base_pos` of both lanes, it can determine `x` location of the center of the lane. Assuming the camera is mounted in the middle of the car, the distance between this `x` location to pixel `x=1280//2` would be the result. Then it can be converted to real world distance using `Camera.xm_per_pix`.

```python
text = "Distance from lane center: {:.1f} (m)".format(((self.left_lane.best_line_base_pos+self.right_lane.best_line_base_pos)/2 - 960)*self.camera.xm_per_pix)
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #532 through #536 in my code in `adv_lane_finding.py` in `LaneFinding.process_image()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the issues I faced is find the good tradeoff between using saturation threshold(in HSL color space) and Value threshold(in LSV color space). Saturation does a really good job on most of the frames, but struggles in shadow. On the other hand, value threshold excels in shadow, but can't detect anything on bright light. After a few trail and error I decided to use saturation and gradient in combination to generate a work on all scenario mask but sacrifice some of the performance provided by just saturation threshold. To compensate that, I will have to implement some filtering and ignore frames that cannot detect lanes. 

Another issue came up during testing is that it will detect the edges on the curb instead of actual lane mark in some of the frames where lane mark is not obvious. I have to implement a sanity check of the line base position so that it filters out those mistakenly detected frames.

Finally, I think in general this has consistent performance. But the polynomial fit on the dotted lane mark is still jumpping around after filtering. I'd like some advice on how I can improve that. And my algorithm is struggling on light color surface like concrete. I believe it is due to my combined filter relies on both saturation and gradient at the same time. It'd be nice to have a way to use saturation only in certain frames and use gradient on other frames.
