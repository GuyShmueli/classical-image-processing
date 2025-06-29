import numpy as np
import mediapy as mp


def color2gray(video):
    """
    Convert an RGB video to 8-bit grayscale.
    The video array is expected to be of shape (num_frames, height, width, 3).
    Broadcasting applies the grayscale conversion to all frames simultaneously.
    """

    # The np.sum function is used in the grayscale conversion process because it effectively combines the weighted
    # contributions of the Red, Green, and Blue channels into a single intensity value per pixel.

    # Weights for the RGB channels
    # Reshape the weights to be (1, 1, 1, 3) for broadcasting
    weights = np.array([0.2989, 0.5870, 0.1140]).reshape((1, 1, 1, 3))

    # axis=-1 goes to the last dimension (channels here) and summing over the channels dimension,
    # effectively reducing this dimension (and indeed greyscale video has (num_frame, height, width), without channels).
    video = np.sum(video * weights, axis=-1).astype(np.uint8)
    return video


def calc_frame_hist_and_cum_hist(video):
    """ Compute per-frame 256-bin histograms and their cumulative sums. """
    frame_hist = []
    frame_cum_hist = []
    for frame in video:
        # calculating histogram
        hist = np.histogram(frame, bins=256, range=[0, 256])[0]
        frame_hist.append(hist)

        # calculating cumulative histogram
        cum_hist = (np.cumsum(hist))
        frame_cum_hist.append(cum_hist)

    return frame_hist, frame_cum_hist

def consecutive_cum_hist_distance_arr(cum_hist_frames_arr):
    """
    Return an array whose i-th element is the absolute difference between
    the *total* cumulative-histogram sums of frames i and i+1.
    """
    dist = []
    for index, cum_hist in enumerate(cum_hist_frames_arr[:-1]):
        next_cum_hist = cum_hist_frames_arr[index + 1].astype(np.int64)
        cum_hist = cum_hist.astype(np.int64)
        difference = np.abs(np.sum(next_cum_hist) - np.sum(cum_hist))
        dist.append(difference)
    return np.asarray(dist)


def main(video_path, video_type):
    """
    Detect a single hard scene cut in the video.
    Returns
    -------
    (k, k+1)
        Index of the last frame of the first scene and the first frame of
        the second scene.
    """
    vid = mp.read_video(video_path)
    gray_vid = color2gray(vid)
    cum_hist_gray_vid_frames = calc_frame_hist_and_cum_hist(gray_vid)[1]
    diffs = consecutive_cum_hist_distance_arr(cum_hist_gray_vid_frames)
    max_diff = diffs.argmax()
    return (max_diff, max_diff+1)
