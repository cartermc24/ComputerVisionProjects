#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is a python reimplementation of the open source tracker in
http://www2.isr.uc.pt/~henriques/circulant/index.html

Found http://wiki.scipy.org/NumPy_for_Matlab_Users very useful

Based on the work of João F. Henriques, 2012
http://www.isr.uc.pt/~henriques

Rodrigo Benenson, MPI-Inf 2013
http://rodrigob.github.io
"""

from __future__ import print_function

import os
import os.path
import sys
import glob
import time
from optparse import OptionParser
from copy import deepcopy
from numpy import mean, std

import scipy.misc
import pylab

debug = False

class CirculantMatrixTracker:

    def __init__(self, object_example):
        """
        object_example is an image showing the object to track
        """

        return

    def find(self, image):
        """
        Will return the x/y coordinates where the object was found,
        and the score
        """

        return

    def update_template(self, new_example, forget_factor=1):
        """
        Update the tracking template,
        new_example is expected to match the size of
        the example provided to the constructor
        """

        return


def load_video_info(video_path):
    """
    Loads all the relevant information for the video in the given path:
    the list of image files (cell array of strings), initial position
    (1x2), target size (1x2), whether to resize the video to half
    (boolean), and the ground truth information for precision calculations
    (Nx2, for N frames). The ordering of coordinates is always [y, x].

    The path to the video is returned, since it may change if the images
    are located in a sub-folder (as is the default for MILTrack's videos).
    """

    # load ground truth from text file (MILTrack's format)
    text_files = glob.glob(os.path.join(video_path, "*_gt.txt"))
    assert text_files, \
        "No initial position and ground truth (*_gt.txt) to load."

    first_file_path = os.path.join(video_path, text_files[0])
    #f = open(first_file_path, "r")
    #ground_truth = textscan(f, '%f,%f,%f,%f') # [x, y, width, height]
    #ground_truth = cat(2, ground_truth{:})
    ground_truth = pylab.loadtxt(first_file_path, delimiter=",")
    #f.close()

    # set initial position and size
    first_ground_truth = ground_truth[0, :]
    # target_sz contains height, width
    target_sz = pylab.array([first_ground_truth[3], first_ground_truth[2]])
    # pos contains y, x center
    pos = [first_ground_truth[1], first_ground_truth[0]] \
        + pylab.floor(target_sz / 2)
    
    #try:
    if True:
        # interpolate missing annotations
        # 4 out of each 5 frames is filled with zeros
        for i in range(4):  # x, y, width, height
            xp = range(0, ground_truth.shape[0], 5)
            fp = ground_truth[xp, i]
            x = range(ground_truth.shape[0])
            ground_truth[:, i] = pylab.interp(x, xp, fp)
        # store positions instead of boxes
        ground_truth = ground_truth[:, [1, 0]] + ground_truth[:, [3, 2]] / 2
    #except Exception as e:
    else:
        print("Failed to gather ground truth data")
        #print("Error", e)
        # ok, wrong format or we just don't have ground truth data.
        ground_truth = []

    # Get MILTrack results
    mil_files = glob.glob(os.path.join(video_path, "*_TR001.txt"))
    assert mil_files, \
        "No MILTrack results (*_TR001.txt) to load."
    mil_file_path = os.path.join(video_path, mil_files[0])
    mil_results = pylab.loadtxt(mil_file_path, delimiter=",")
    #first_mil_result = mil_results[0, :]
    #mil_sz = pylab.array([first_mil_result[3], first_mil_result[2]])
    #mil_pos = [first_mil_result[1], first_mil_result[0]]

    # list all frames. first, try MILTrack's format, where the initial and
    # final frame numbers are stored in a text file. if it doesn't work,
    # try to load all png/jpg files in the folder.

    text_files = glob.glob(os.path.join(video_path, "*_frames.txt"))
    if text_files:
        first_file_path = os.path.join(video_path, text_files[0])
        #f = open(first_file_path, "r")
        #frames = textscan(f, '%f,%f')
        frames = pylab.loadtxt(first_file_path, delimiter=",", dtype=int)
        #f.close()

        # see if they are in the 'imgs' subfolder or not
        test1_path_to_img = os.path.join(video_path,
                                         "imgs/img%05i.png" % frames[0])
        test2_path_to_img = os.path.join(video_path,
                                         "img%05i.png" % frames[0])
        if os.path.exists(test1_path_to_img):
            video_path = os.path.join(video_path, "imgs/")
        elif os.path.exists(test2_path_to_img):
            video_path = video_path  # no need for change
        else:
            raise Exception("Failed to find the png images")

        # list the files
        img_files = ["img%05i.png" % i
                     for i in range(frames[0], frames[1] + 1)]
        #img_files = num2str((frames{1} : frames{2})', 'img%05i.png')
        #img_files = cellstr(img_files);
    else:
        # no text file, just list all images
        img_files = glob.glob(os.path.join(video_path, "*.png"))
        if len(img_files) == 0:
            img_files = glob.glob(os.path.join(video_path, "*.jpg"))

        assert len(img_files), "Failed to find png or jpg images"

        img_files.sort()

    # if the target is too large, use a lower resolution
    # no need for so much detail
    if pylab.sqrt(pylab.prod(target_sz)) >= 100:
        pos = pylab.floor(pos / 2)
        target_sz = pylab.floor(target_sz / 2)
        resize_image = True
    else:
        resize_image = False

    ret = [img_files, pos, target_sz, resize_image, ground_truth, video_path,
            mil_results]
    return ret


def rgb2gray(rgb_image):
    "Based on http://stackoverflow.com/questions/12201577"
    # [0.299, 0.587, 0.144] normalized gives [0.29, 0.57, 0.14]
    return pylab.dot(rgb_image[:, :, :3], [0.29, 0.57, 0.14])


def get_subwindow(im, pos, sz, cos_window):
    """
    Obtain sub-window from image, with replication-padding.
    Returns sub-window of image IM centered at POS ([y, x] coordinates),
    with size SZ ([height, width]). If any pixels are outside of the image,
    they will replicate the values at the borders.

    The subwindow is also normalized to range -0.5 .. 0.5, and the given
    cosine window COS_WINDOW is applied
    (though this part could be omitted to make the function more general).
    """

    if pylab.isscalar(sz):  # square sub-window
        sz = [sz, sz]

    ys = pylab.floor(pos[0]) \
        + pylab.arange(sz[0], dtype=int) - pylab.floor(sz[0]/2)
    xs = pylab.floor(pos[1]) \
        + pylab.arange(sz[1], dtype=int) - pylab.floor(sz[1]/2)

    ys = ys.astype(int)
    xs = xs.astype(int)

    # check for out-of-bounds coordinates,
    # and set them to the values at the borders
    ys[ys < 0] = 0
    ys[ys >= im.shape[0]] = im.shape[0] - 1

    xs[xs < 0] = 0
    xs[xs >= im.shape[1]] = im.shape[1] - 1
    #zs = range(im.shape[2])

    # extract image
    #out = im[pylab.ix_(ys, xs, zs)]
    out = im[pylab.ix_(ys, xs)]

    if debug:
        print("Out max/min value==", out.max(), "/", out.min())
        pylab.figure()
        pylab.imshow(out, cmap=pylab.cm.gray)
        pylab.title("cropped subwindow")

    #pre-process window --
    # normalize to range -0.5 .. 0.5
    # pixels are already in range 0 to 1
    out = out.astype(pylab.float64) - 0.5

    # apply cosine window
    out = pylab.multiply(cos_window, out)

    return out


def dense_gauss_kernel(sigma, x, y=None):
    """
    Gaussian Kernel with dense sampling.
    Evaluates a gaussian kernel with bandwidth SIGMA for all displacements
    between input images X and Y, which must both be MxN. They must also
    be periodic (ie., pre-processed with a cosine window). The result is
    an MxN map of responses.

    If X and Y are the same, ommit the third parameter to re-use some
    values, which is faster.
    """

    xf = pylab.fft2(x)  # x in Fourier domain
    x_flat = x.flatten()
    xx = pylab.dot(x_flat.transpose(), x_flat)  # squared norm of x

    if y is not None:
        # general case, x and y are different
        yf = pylab.fft2(y)
        y_flat = y.flatten()
        yy = pylab.dot(y_flat.transpose(), y_flat)
    else:
        # auto-correlation of x, avoid repeating a few operations
        yf = xf
        yy = xx

    # cross-correlation term in Fourier domain
    xyf = pylab.multiply(xf, pylab.conj(yf))

    # to spatial domain
    xyf_ifft = pylab.ifft2(xyf)
    #xy_complex = circshift(xyf_ifft, floor(x.shape/2))
    row_shift, col_shift = pylab.floor(pylab.array(x.shape)/2).astype(int)
    xy_complex = pylab.roll(xyf_ifft, row_shift, axis=0)
    xy_complex = pylab.roll(xy_complex, col_shift, axis=1)
    xy = pylab.real(xy_complex)

    # calculate gaussian response for all positions
    scaling = -1 / (sigma**2)
    xx_yy = xx + yy
    xx_yy_2xy = xx_yy - 2 * xy
    k = pylab.exp(scaling * pylab.maximum(0, xx_yy_2xy / x.size))

    #print("dense_gauss_kernel x.shape ==", x.shape)
    #print("dense_gauss_kernel k.shape ==", k.shape)

    return k


def show_precision(positions, ground_truth, video_path, title):
    """
    Calculates precision for a series of distance thresholds (percentage of
    frames where the distance to the ground truth is within the threshold).
    The results are shown in a new figure.

    Accepts positions and ground truth as Nx2 matrices (for N frames), and
    a title string.
    """
    #print(positions)
    print("Evaluating tracking results.")

    pylab.ioff()  # interactive mode off

    max_threshold = 50  # used for graphs in the paper

    if positions.shape[0] != ground_truth.shape[0]:
        raise Exception(
            "Could not plot precisions, because the number of ground"
            "truth frames does not match the number of tracked frames.")

    # calculate distances to ground truth over all frames
    delta = positions - ground_truth
    distances = pylab.sqrt((delta[:, 0]**2) + (delta[:, 1]**2))
    #distances[pylab.isnan(distances)] = []

    # compute precisions
    precisions = pylab.zeros((max_threshold, 1), dtype=float)
    for p in range(max_threshold):
        precisions[p] = pylab.sum(distances <= p, dtype=float) / len(distances)

    if False:
        pylab.figure()
        pylab.plot(distances)
        pylab.title("Distances")
        pylab.xlabel("Frame number")
        pylab.ylabel("Distance")

    # plot the precisions
    pylab.figure()  # 'Number', 'off', 'Name',
    pylab.title("Precisions - " + title)
    pylab.plot(precisions, "k-", linewidth=2)
    pylab.xlabel("Threshold")
    pylab.ylabel("Precision")
    pylab.ylim(top=1.1)
    #pylab.ylim(bottom=0.0)    
    pylab.show()
    return


def show_psr_plot(psr_values, title):
    pylab.figure()
    pylab.title("PSR Values - " + title)
    pylab.plot(psr_values)
    pylab.xlabel("Frame number")
    pylab.ylabel("PSR")
    pylab.show()
    return


def plot_tracking(frame, pos, target_sz, im, ground_truth, mil_results, mil_pos):
    
    # Get MIL values
    mil_frame_results = mil_results[frame]
    mil_frame_sz = pylab.array([mil_frame_results[3], mil_frame_results[2]])
    mil_frame_pos = pylab.array([mil_frame_results[1], mil_frame_results[0]] + pylab.floor(mil_frame_sz / 2))
    mil_pos[frame] = mil_frame_pos

    global \
        tracking_figure, tracking_figure_title, tracking_figure_axes, \
        tracking_rectangle, gt_point, \
        z_figure_axes, response_figure_axes, mil_rectangle

    timeout = 1e-6
    #timeout = 0.05  # uncomment to run slower
    if frame == 0:
        #pylab.ion()  # interactive mode on
        tracking_figure = pylab.figure()
        gs = pylab.GridSpec(1, 3, width_ratios=[3, 1, 1])

        tracking_figure_axes = tracking_figure.add_subplot(gs[0])
        tracking_figure_axes.set_title("Tracked object (and ground truth)")

        z_figure_axes = tracking_figure.add_subplot(gs[1])
        z_figure_axes.set_title("Template")

        response_figure_axes = tracking_figure.add_subplot(gs[2])
        response_figure_axes.set_title("Response")

        tracking_rectangle = pylab.Rectangle((0, 0), 0, 0)
        tracking_rectangle.set_color((1, 0, 0, 0.5)) # Red
        tracking_figure_axes.add_patch(tracking_rectangle)
        
        # Add MIL rectangle
        mil_rectangle = pylab.Rectangle((0, 0), 0, 0)
        mil_rectangle.set_color((0, 1, 0, 0.5)) # Green
        tracking_figure_axes.add_patch(mil_rectangle)

        gt_point = pylab.Circle((0, 0), radius=5)
        gt_point.set_color((0, 0, 1, 0.5))
        tracking_figure_axes.add_patch(gt_point)

        tracking_figure_title = tracking_figure.suptitle("")

        pylab.show(block=False)

    elif tracking_figure is None:
        return  # we simply go faster by skipping the drawing
    elif not pylab.fignum_exists(tracking_figure.number):
        #print("Drawing window closed, end of game. "
        #      "Have a nice day !")
        #sys.exit()
        print("From now on drawing will be omitted, "
              "so that computation goes faster")
        tracking_figure = None
        return

    global z, response
    tracking_figure_axes.imshow(im, cmap=pylab.cm.gray)
    rect_y, rect_x = tuple(pos - target_sz/2.0)
    rect_height, rect_width = target_sz
    tracking_rectangle.set_xy((rect_x, rect_y))
    tracking_rectangle.set_width(rect_width)
    tracking_rectangle.set_height(rect_height)

    # Update MIL Rectangle
    mil_rect_y, mil_rect_x = tuple(mil_frame_pos - mil_frame_sz / 2.0)
    mil_rect_height, mil_rect_width = mil_frame_sz
    mil_rectangle.set_xy((mil_rect_x, mil_rect_y))
    mil_rectangle.set_width(mil_rect_width)
    mil_rectangle.set_height(mil_rect_height)

    if len(ground_truth) > 0:
        gt = ground_truth[frame]
        gt_y, gt_x = gt
        gt_point.center = (gt_x, gt_y)

    if z is not None:
        z_figure_axes.imshow(z, cmap=pylab.cm.hot)

    if response is not None:
        response_figure_axes.imshow(response, cmap=pylab.cm.hot)

    tracking_figure_title.set_text("Frame %i (out of %i)"
                                   % (frame + 1, len(ground_truth)))

    if debug and False and (frame % 1) == 0:
        print("Tracked pos ==", pos)

    #tracking_figure.canvas.draw()  # update
    pylab.draw()
    pylab.waitforbuttonpress(timeout=timeout)

    return


def track(input_video_path, psr_threshold):
    """
    notation: variables ending with f are in the frequency domain.
    """
    output_file = open('results.txt', 'w')

    # parameters according to the paper --
    padding = 1.0  # extra area surrounding the target
    #spatial bandwidth (proportional to target)
    output_sigma_factor = 1 / float(16)
    sigma = 0.2  # gaussian kernel bandwidth
    lambda_value = 1e-2  # regularization
    # linear interpolation factor for adaptation
    interpolation_factor = 0.075

    info = load_video_info(input_video_path)
    img_files, pos, target_sz, \
        should_resize_image, ground_truth, video_path, mil_results = info

    # window size, taking padding into account
    sz = pylab.floor(target_sz * (1 + padding))

    # desired output (gaussian shaped), bandwidth proportional to target size
    output_sigma = pylab.sqrt(pylab.prod(target_sz)) * output_sigma_factor

    grid_y = pylab.arange(sz[0]) - pylab.floor(sz[0]/2)
    grid_x = pylab.arange(sz[1]) - pylab.floor(sz[1]/2)
    #[rs, cs] = ndgrid(grid_x, grid_y)
    rs, cs = pylab.meshgrid(grid_x, grid_y)
    y = pylab.exp(-0.5 / output_sigma**2 * (rs**2 + cs**2))
    yf = pylab.fft2(y)
    #print("yf.shape ==", yf.shape)
    #print("y.shape ==", y.shape)

    # store pre-computed cosine window
    cos_window = pylab.outer(pylab.hanning(sz[0]),
                             pylab.hanning(sz[1]))

    total_time = 0  # to calculate FPS
    positions = pylab.zeros((len(img_files), 2))  # to calculate precision

    global z, response
    z = None
    alphaf = None
    response = None

    # Occlusion vars
    psr_values = [0] * len(img_files)

    # MIL positions
    mil_positions = pylab.zeros((len(img_files), 2)) 

    for frame, image_filename in enumerate(img_files):

        if True and ((frame % 10) == 0):
            print("Processing frame", frame)

        # load image
        image_path = os.path.join(video_path, image_filename)
        im = pylab.imread(image_path)
        if len(im.shape) == 3 and im.shape[2] > 1:
            im = rgb2gray(im)

        #print("Image max/min value==", im.max(), "/", im.min())

        if should_resize_image:
            im = scipy.misc.imresize(im, 0.5)

        start_time = time.time()

        # extract and pre-process subwindow
        x = get_subwindow(im, pos, sz, cos_window)

        is_first_frame = (frame == 0)

        if not is_first_frame:
            # calculate response of the classifier at all locations
            k = dense_gauss_kernel(sigma, x, z)
            kf = pylab.fft2(k)
            alphaf_kf = pylab.multiply(alphaf, kf)
            response = pylab.real(pylab.ifft2(alphaf_kf))  # Eq. 9
            
            #print(response.shape)

            # target location is at the maximum response
            r = response
            row, col = pylab.unravel_index(r.argmax(), r.shape)
            pos = pos - pylab.floor(sz/2) + [row, col]
            
           
           # Occlusion detection
            psr = calculate_psr(response, row, col)
            psr_values[frame] = psr

            is_occlusion = (psr_threshold and psr < psr_threshold)
            if is_occlusion:
                # Occlusion detected
                print("Occlusion detected")
                output_file.write("Occlusion Detected\n")

                # Calculate velocity and predict next location
                # Need at least two frames
                if frame > 2:
                    y_velocity = positions[frame-1][0] - positions[frame-2][0]
                    x_velocity = positions[frame-1][1] - positions[frame-2][1]
                    y_pred = positions[frame-1][0] + y_velocity
                    x_pred = positions[frame-1][1] + x_velocity
                    row,col = y_pred, x_pred

            if debug:
                print("Frame ==", frame)
                print("Max response", r.max(), "at", [row, col])
                print("PSR = {}".format(psr))
                pylab.figure()
                pylab.imshow(cos_window)
                pylab.title("cos_window")

                pylab.figure()
                pylab.imshow(x)
                pylab.title("x")

                pylab.figure()
                pylab.imshow(response)
                pylab.title("response")
                pylab.show(block=True)

        # end "if not first frame"
        output_file.write("{},{},{},{}\n".format(pos[1], pos[0], sz[1], sz[0]))

        # get subwindow at current estimated target position,
        # to train classifer
        x = get_subwindow(im, pos, sz, cos_window)

        # Kernel Regularized Least-Squares,
        # calculate alphas (in Fourier domain)
        k = dense_gauss_kernel(sigma, x)
        new_alphaf = pylab.divide(yf, (pylab.fft2(k) + lambda_value))  # Eq. 7
        new_z = x

        if is_first_frame:
            #first frame, train with a single image
            alphaf = new_alphaf
            z = x
        elif not is_occlusion:
            # subsequent frames, interpolate model
            f = interpolation_factor
            alphaf = (1 - f) * alphaf + f * new_alphaf
            z = (1 - f) * z + f * new_z
        # end "first frame or not"

        # save position and calculate FPS
        positions[frame, :] = pos
        total_time += time.time() - start_time

        # visualization
        plot_tracking(frame, pos, target_sz, im, ground_truth, mil_results, mil_positions)
    # end of "for each image in video"

    if should_resize_image:
        positions = positions * 2

    print("Frames-per-second:",  len(img_files) / total_time)

    title = os.path.basename(os.path.normpath(input_video_path))

    if len(ground_truth) > 0:
        # show the precisions plot
        show_precision(positions, ground_truth, video_path, title)

    if len(mil_results) > 0:
        show_precision(mil_positions, ground_truth, video_path, title + " MIL Result")

    # Show the PSR plot
    show_psr_plot(psr_values, title)

    return

# max_row and max_col represent the maximum response value
def calculate_psr(orig_response, max_row, max_col):
    response = deepcopy(orig_response)
    
    lobe_sigma = 0
    lobe_mean = 0

    num_rows, num_cols = response.shape
    #print("Num rows = {}, num cols = {}".format(num_rows, num_cols))

    if num_rows < 11 or num_cols < 11:
        print("ERROR: response not large enough for PSR calculation")
        return -1

    edge_size = (11 - 1) // 2
  
    # Get only the sidelobe values
    for row in range(max_row - edge_size, max_row + edge_size + 1):
        for col in range(max_col - edge_size, max_col + edge_size + 1):
            response[row][col] = 0
    response = response.flatten()
    response.sort()
    response = response[11**2:]

    lobe_sigma = std(response)
    lobe_mean = mean(response)           
 

    """
    total = 0
    count = 0
   
    for row in range(0, num_rows):
        for col in range(0, num_cols):
            # Ignore max region
            if (row >= (max_row - edge_size) and row <= (max_row + edge_size) and
            col >= (max_col - edge_size) and col <= (max_col + edge_size)):
                continue
            else:
                total += response[row][col]
                count += 1

    #print("count = {}".format(count))
    lobe_mean = total / count

    # Calculate sigma
    for row in range(0, num_rows):
        for col in range(0, num_cols):
            # Ignore max region
            if (row >= (max_row - edge_size) and row <= (max_row + edge_size) and
            col >= (max_col - edge_size) and col <= (max_col + edge_size)):
                continue
            else:
                lobe_sigma += (response[row][col] - lobe_mean)**2

    lobe_sigma /= count
    """

    g_max = orig_response[max_row][max_col]
    result = (g_max - lobe_mean) / lobe_sigma

    return result




def parse_arguments():

    parser = OptionParser()
    parser.description = \
        "This program will track objects " \
        "on videos in the MILTrack paper format. " \
        "See http://goo.gl/pSTo9r"

    parser.add_option("-i", "--input", dest="video_path",
                      metavar="PATH", type="string", default=None,
                      help="path to a folder o a MILTrack video")

    parser.add_option("-t", "--threshold", dest="psr_threshold", type="float",
            default=None, help="Threshold for PSR to detect occlusion")

    (options, args) = parser.parse_args()
    #print (options, args)

    if not options.video_path:
        parser.error("'input' option is required to run this program")
    if not os.path.exists(options.video_path):
            parser.error("Could not find the input file %s"
                         % options.video_path)

    return options


def main():
    options = parse_arguments()

    track(options.video_path, options.psr_threshold)

    print("End of game, have a nice day!")
    return


if __name__ == "__main__":

    main()

# end of file
