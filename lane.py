import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import pickle
import os
from moviepy.editor import VideoFileClip

class Lane():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # y values
        self.plot_y = None
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # polynomial coefficients for last 5 iterations
        self.all_fit = []
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = 0
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

        self.n = 3

        self.margin = 100

        self.img_shape = (1280,720)

        self.lane_inds = np.array([])

    def lane_curve_radius(self):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 800  # meters per pixel in x dimension

        y_eval = np.max(self.plot_y)
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(self.plot_y * ym_per_pix, self.bestx * xm_per_pix, 2)

        # Calculate the new radii of curvature
        curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * fit_cr[0])

        # Now our radius of curvature is in meters

        self.line_base_pos = abs(self.bestx[-1] - self.img_shape[0] / 2.) * xm_per_pix

        return curverad

    def initialize(self, img, x, ploty, poly_fit, lane_inds):
        self.recent_xfitted.insert(0,x)
        self.plot_y = ploty
        self.bestx = x
        self.all_fit.insert(0,poly_fit)
        self.current_fit = self.all_fit[0]
        self.best_fit = self.current_fit
        self.radius_of_curvature = self.lane_curve_radius()
        self.lane_inds = lane_inds

        self.detected = True

        if len(x) == 0 or len(ploty) == 0:
            self.detected = False



    def update(self, nonzero):

        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        lane_inds = (
        (nonzerox > (self.best_fit[0] * (nonzeroy ** 2) + self.best_fit[1] * nonzeroy + self.best_fit[2] - self.margin)) & (
        nonzerox < (self.best_fit[0] * (nonzeroy ** 2) + self.best_fit[1] * nonzeroy + self.best_fit[2] + self.margin)))

        # Again, extract left and right line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        if len(x) == 0 or len(y) == 0:
            self.detected = False
            return

        self.lane_inds = lane_inds
        # Fit a second order polynomial to each
        fit = np.polyfit(y, x, 2)

        self.current_fit = fit
        self.detected = True
        # Generate x and y values for plotting
        # ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        fitx = fit[0] * self.plot_y ** 2 + fit[1] * self.plot_y + fit[2]


        if len(self.recent_xfitted) >= self.n:
            self.recent_xfitted.pop(-1)

        if len(self.all_fit) >= self.n:
            self.all_fit.pop(-1)

        self.all_fit.insert(0,fit)
        self.recent_xfitted.insert(0,fitx)
        weights = [0.5,0.3,0.2]
        if len(self.recent_xfitted) < self.n or len(self.all_fit) < self.n:
            self.bestx = self.recent_xfitted[0]
            self.best_fit = self.all_fit[0]
        else:
            self.bestx = np.average(np.array(self.recent_xfitted), axis=0,weights=weights)
            self.best_fit = np.average(np.array(self.all_fit),axis=0, weights=weights)

        self.radius_of_curvature = self.lane_curve_radius()



class LaneFinder:

    # number of chessboard corners
    nx = 9
    ny = 6
    # camera calibration matrix
    mtx = None
    #distortion coeffs
    dist = None
    #pickle storage for above values
    dist_pickle = {}

    # left detected lane
    left_lane = None
    # right detected lane
    right_lane = None

    #perspective transform matrix
    perspective_matrix = np.array([False])

    #source points for warping (trapezoid around lane)
    src_points = None
    #destination points for warping (square)
    dst_points = None
    # flag whether to save pipeline snapshots
    save_images = True




    #Step 1
    def calibrate_camera(self,calib_path='./camera_cal/', output_path='./output_images/'):

        ny = self.ny
        nx = self.nx

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((ny * nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob(calib_path + 'calibration*.jpg')
        img_size = None

        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            img_size = (img.shape[1],img.shape[0])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        return mtx,dist

    def undistort(self,img):
        dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return dst

    #Step 2
    def abs_sobel_thresh(self,image, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Calculate directional gradient
        # Apply threshold
        x = 0
        y = 0
        if orient == 'x':
            x = 1
        elif orient == 'y':
            y = 1

        sobel = cv2.Sobel(image, cv2.CV_64F, x, y, ksize=sobel_kernel)
        s_abs = np.absolute(sobel)
        scaled_sobel = np.uint8(255. * s_abs / np.max(s_abs))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return sxbinary

    def mag_thresh(self,image, sobel_kernel=3, mag_thresh=(0, 255)):
        # Calculate gradient magnitude
        # Apply threshold

        sx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        s_mag = np.sqrt(sx ** 2 + sy ** 2)
        scaled_sobel = np.uint8(255. * (s_mag / np.max(s_mag)))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
        return sxbinary

    def dir_threshold(self,image, sobel_kernel=3, thresh=(0, np.pi / 2)):
        # Calculate gradient direction
        # Apply threshold
        sx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        dir = np.arctan2(np.absolute(sy), np.absolute(sx))

        sxbinary = np.zeros_like(dir)
        sxbinary[(dir >= thresh[0]) & (dir <= thresh[1])] = 1
        return sxbinary

    #Step 3
    def grad_threshold(self,gray):
        # Choose a Sobel kernel size
        ksize = 5  # Choose a larger odd number to smooth gradient measurements

        # Apply each of the thresholding functions
        gradx = self.abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(50, 255))
        grady = self.abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(50, 255))
        mag_binary = self.mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(50, 255))
        dir_binary = self.dir_threshold(gray, sobel_kernel=ksize, thresh=(np.pi / 4, np.pi / 2))

        combined = np.zeros_like(dir_binary)
        # combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        combined[((gradx == 1) | (grady == 1)) & (dir_binary == 1) & ((mag_binary == 1))] = 1

        scaled_combined = np.uint8(255.*combined)
        return scaled_combined

    def color_threshold(self, img, thresh = (100,255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        s_channel = hls[:, :, 2]
        l_channel = hls[:,:,1]
        h_channel = hls[:,:,0]

        # h_channel = hls[0,:,:]
        # return h_channel
        # Threshold color channel
        s_thresh_min = thresh[0]
        s_thresh_max = thresh[1]
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max) & (l_channel >= 75)] = 1


        scaled_binary = np.uint8(255. * s_binary)

        return scaled_binary

    def combined_threshold(self,color_thresholded,grad_thresholded):
        combined_threshold = np.zeros_like(color_thresholded)
        combined_threshold[(grad_thresholded == 255) | (color_thresholded == 255)] = 255

        color_binary = np.dstack((grad_thresholded, color_thresholded, np.zeros_like(color_thresholded)))

        return combined_threshold,color_binary

    #Step 4
    def birds_eye_view(self, gray, file_name):

        # tl = (574, 457)
        # tr = (704, 457)
        # br = (1040, 660)
        # bl = (290, 660)

        # tl = (554, 457)
        # tr = (724, 457)
        # br = (1180, 720)
        # bl = (100, 720)
        midpoint = int(gray.shape[1]/2)
        car_hood_width_div_2 = int(1280 / 2)
        top_length = int(car_hood_width_div_2 * 0.26)
        top_x = midpoint - top_length
        x_offset = (midpoint - int(top_length/2)) - (midpoint - car_hood_width_div_2)
        y_offset = 225
        # bot_length = 1080

        bl = (midpoint - car_hood_width_div_2,680)
        tl = (bl[0] + x_offset, bl[1] - y_offset)
        tr = (tl[0] + top_length, tl[1])
        br = (tr[0] + x_offset, bl[1])
        # tl = (554, 485)
        # tr = (814, 485)
        # br = (1230, 720)
        # bl = (150, 720)

        debug_img = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        cv2.line(debug_img,tl,tr,(0,0,255),1)
        cv2.line(debug_img, tr, br, (0, 0, 255), 1)
        cv2.line(debug_img, br, bl, (0, 0, 255), 1)
        cv2.line(debug_img, bl, tl, (0, 0, 255), 1)

        rect_t = [tl,tr,br,bl]
        rect = np.array(rect_t,dtype="float32")

        dst_t = [[0,0], [1280,0],[1280,720],[0,720]]
        dst = np.array(dst_t, dtype="float32")

        self.src_points = rect
        self.dst_points = dst

        self.perspective_matrix = cv2.getPerspectiveTransform(rect, dst)

        warped = cv2.warpPerspective(gray, self.perspective_matrix, (1280,720))


        # for test/debug purposes
        self.save_frame(debug_img, img_path=file_name + "_birds_eye_debug.jpg")
        warped_birds_eye_debug = cv2.warpPerspective(debug_img, self.perspective_matrix, (1280,720))
        self.save_frame(warped_birds_eye_debug, img_path=file_name + "_birds_eye_debug_warped.jpg")

        return warped

    #Step 5
    def detect_lanes_using_current(self, img):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = img.nonzero()
        nonzeroy = nonzero[0]
        nonzerox = nonzero[1]
        self.left_lane.update(nonzero)
        self.right_lane.update(nonzero)

        if self.left_lane.detected == False or self.right_lane.detected == False:
            return

        if self.is_valid_lane_lines(self.left_lane,self.right_lane) == False:
            self.left_lane.detected = False
            self.right_lane.detected = False
            return

        out_img = np.dstack((img, img, img))

        out_img[nonzeroy[self.left_lane.lane_inds], nonzerox[self.left_lane.lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[self.right_lane.lane_inds], nonzerox[self.right_lane.lane_inds]] = [0, 0, 255]

        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_fitx = self.left_lane.bestx
        right_fitx = self.right_lane.bestx
        margin = self.left_lane.margin
        ploty = self.left_lane.plot_y

        margin = self.left_lane.margin
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (255, 255, 255))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (255, 255, 255))

        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        return result

    def is_valid_lane_lines(self,left_lane:Lane,right_lane:Lane):
        # pick 3 points on each line (top, middle, bottom) and confirm the distance between each corresponding point of lines is roughly equal
        left_line = np.transpose(np.vstack([left_lane.bestx, left_lane.plot_y]))
        right_line = np.transpose(np.vstack([right_lane.bestx, right_lane.plot_y]))
        left_line_points = np.array([left_line[0],left_line[int(len(left_line)/2)], left_line[len(left_line)-1]])
        right_line_points = np.array([right_line[0], right_line[int(len(right_line) / 2)], right_line[len(right_line) - 1]])
        diff = right_line_points - left_line_points
        delta_mean = abs(np.mean(diff, axis=0))

        left_curve = abs(left_lane.lane_curve_radius())
        right_curve = abs(right_lane.lane_curve_radius())

        return delta_mean[0] >= 790 and delta_mean[0] <= 830 #and abs(left_curve - right_curve) / left_curve <= 0.25


    def detect_lanes_from_scratch(self, img, file_name):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[int(img.shape[0] / 2):, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((img, img, img))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(img.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 100
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        total_left_lane_delta = 0
        total_right_lane_delta = 0

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                mean_left_x = np.int(np.mean(nonzerox[good_left_inds]))
                # total_left_lane_delta += (mean_left_x - leftx_current)

                leftx_current = mean_left_x
            else:
                leftx_current += int(total_left_lane_delta / max(window,1))
            if len(good_right_inds) > minpix:
                mean_right_x = np.int(np.mean(nonzerox[good_right_inds]))
                # total_right_lane_delta += (mean_right_x - rightx_current)
                rightx_current = mean_right_x
            else:
                rightx_current += int(total_right_lane_delta / max(window,1))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        if len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0:
            return None

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        self.left_lane = Lane()
        self.right_lane = Lane()

        self.left_lane.initialize(img,left_fitx,ploty,left_fit, left_lane_inds)
        self.right_lane.initialize(img, right_fitx, ploty, right_fit, right_lane_inds)

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        self.save_frame(out_img,file_name + '_histogram.jpg')

        # Create an image to draw on and an image to show the selection window
        # out_img = np.dstack((img, img, img))
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        margin = 1
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 255))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 255))

        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        self.save_frame(result, file_name + '_histogram_2.jpg')

        return result



    #Step 6
    def unwarp_lanes(self, img):
        # Create an image to draw the lines on
        color_warp = np.zeros(shape=img.shape,dtype=np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_lane.bestx, self.left_lane.plot_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_lane.bestx, self.right_lane.plot_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

        return result


    def combine_images(self, street_image, birds_eye_view, thresholded_image):
        #920,517
        #360,360

        if street_image is None:
            return

        output = np.zeros((720,1280,3),dtype=np.uint8)

        resized_street_image =  cv2.resize(street_image,(920,517))

        # resized_birds_eye_view = cv2.cvtColor(resized_birds_eye_view,cv2.COLOR_GRAY2RGB)
        resized_thresholded_image = cv2.resize(thresholded_image, (360,360))
        output[100:100+resized_street_image.shape[0],0:resized_street_image.shape[1],:] = resized_street_image
        output[0:resized_thresholded_image.shape[0],resized_street_image.shape[1]:,:] = resized_thresholded_image

        if birds_eye_view is not None:
            resized_birds_eye_view = cv2.resize(birds_eye_view, (360, 360))
            output[resized_thresholded_image.shape[0]:,resized_street_image.shape[1]:,:] = resized_birds_eye_view

        leftOrRight = "left of center"
        width_of_lane = 3.6
        offset = (width_of_lane/2.) - self.left_lane.line_base_pos
        if offset < 0:
            leftOrRight = "right of center"
        elif offset == 0:
            leftOrRight = "from the center"

        lane_positions = "Vehicle is {:3.1f}m {direction}".format(abs(offset),direction=leftOrRight)
        curvature = "Radius of curvature:{:5.1f}m".format((self.left_lane.lane_curve_radius() + self.right_lane.lane_curve_radius())/2.)
        cv2.putText(output, lane_positions, (0, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(output, curvature, (0, 90), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255),1)

        return output

    def read_frame(self, frame = None, img_path = None):

        # file name for saving
        img = None
        file_name = ""
        if img_path is not None:
            file_name = img_path.split("/")[-1]
            # load image
            img = cv2.imread(img_path)
        else:
            img = frame

        # undistort image using prior loaded camera matrix + distortion coeffs
        undistorted_img = self.undistort(img)
        self.save_frame(undistorted_img, img_path= file_name + "_undistorted.jpg")
        # gray scale image
        gray = cv2.cvtColor(undistorted_img,cv2.COLOR_BGR2GRAY)
        # use gradient threshold on images using combined thresholds for absolute, magnitude, and directional thresholds
        grad_thresholded = self.grad_threshold(gray)
        self.save_frame(grad_thresholded, img_path= file_name + "_grad_thresholded.jpg")
        # use HLS color thresholding
        color_thresholded = self.color_threshold(img)
        self.save_frame(color_thresholded, img_path=file_name + "_color_thresholded.jpg")

        combined_threshold,color_binary = self.combined_threshold(color_thresholded,grad_thresholded)

        # return color_binary
        self.save_frame(combined_threshold, img_path=file_name + "_combined_threshold.jpg")
        self.save_frame(color_binary, img_path=file_name + "_color_binary.jpg")

        birds_eye = self.birds_eye_view(combined_threshold,file_name=file_name)
        self.save_frame(birds_eye, img_path=file_name + "_birds_eye.jpg")

        warped_lanes = birds_eye
        if self.left_lane is None or self.right_lane is None or self.left_lane.detected is not True or self.right_lane.detected is not True or img_path is not None:
            warped_lanes = self.detect_lanes_from_scratch(birds_eye, file_name)
        else:
            warped_lanes = self.detect_lanes_using_current(birds_eye)
            if self.left_lane.detected is not True or self.right_lane.detected is not True:
                warped_lanes = self.detect_lanes_from_scratch(birds_eye, file_name)
        result = self.unwarp_lanes(img)
        self.save_frame(result, img_path=file_name + "_final.jpg")

        return self.combine_images(result,warped_lanes,color_binary)

    def save_frame(self,img,img_path):
        if self.save_images is True:
            cv2.imwrite('./output_images/' + img_path, img)
    def setup(self):
        try:
            with open("output_images/dist_pickle.p", "rb") as f:
                self.dist_pickle = pickle.load(f)
        except FileNotFoundError:
            self.mtx, self.dist = self.calibrate_camera()
            undistorted_img = self.undistort(cv2.imread('./camera_cal/calibration1.jpg'))
            cv2.imwrite('output_images/undistorted_img.jpg', undistorted_img)
            # save camera matrix and distortion coefficients
            self.dist_pickle["mtx"] = self.mtx
            self.dist_pickle["dist"] = self.dist
            pickle.dump(self.dist_pickle, open("output_images/dist_pickle.p", "wb"))

        self.mtx = self.dist_pickle["mtx"]
        self.dist = self.dist_pickle["dist"]


l = LaneFinder()
l.setup()

debug = True

# clip1 = VideoFileClip("challenge_video.mp4").subclip((0,0),(0,5))
# clip1.save_frame("./test_images/challenge_video_1.jpg",t=0)
# clip1.save_frame("./test_images/challenge_video_2.jpg",t=1)
# clip1.save_frame("./test_images/challenge_video_3.jpg",t=4)

def process_frame(frame):
    return l.read_frame(frame)

if debug is True:
    l.read_frame(img_path="./test_images/straight_lines1.jpg")
    l.read_frame(img_path="./test_images/straight_lines2.jpg")
    l.read_frame(img_path="./test_images/test1.jpg")
    l.read_frame(img_path="./test_images/test2.jpg")

    # l.read_frame(img_path="./test_images/challenge_video_1.jpg")
    # l.read_frame(img_path="./test_images/challenge_video_2.jpg")
    # l.read_frame(img_path="./test_images/challenge_video_3.jpg")

    # output = clip1.fl_image(process_frame)  # NOTE: this function expects color images!!
    # output.write_videofile('challenge_video_processed.mp4', audio=False)

    clip1 = VideoFileClip("project_video.mp4") #.subclip((0, 0), (0, 5))
    clip1.save_frame("./test_images/project_video_1.jpg", t=0)
    clip1.save_frame("./test_images/project_video_2.jpg", t=2)
    clip1.save_frame("./test_images/project_video_3.jpg", t=4)
    l.read_frame(img_path="./test_images/project_video_1.jpg")
    l.read_frame(img_path="./test_images/project_video_2.jpg")
    l.read_frame(img_path="./test_images/project_video_3.jpg")

    # output = clip1.fl_image(process_frame) #NOTE: this function expects color images!!
    # output.write_videofile('project_video_processed.mp4', audio=False)
else:
    l.save_images = False





    # output = clip1.fl_image(process_frame) #NOTE: this function expects color images!!
    # output.write_videofile('project_video_processed.mp4', audio=False)