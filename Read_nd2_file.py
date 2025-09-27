#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 21:02:31 2025

@author: khadijehmasumnia
"""
#%% Import Libraries
import nd2
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import nd2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

#%% Frame_viewer

def frame_viewer(filepath):
    # Open ND2 file
    with nd2.ND2File(filepath) as f:
        data = f.asarray().swapaxes(0,1)
        
        print("Data shape:", data.shape)  
        # expected: (n_videos, n_frames, height, width)

    n_videos, n_frames, _, _ = data.shape

    # Start with first video + first frame
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)  # space for 2 sliders
    img = ax.imshow(data[0, 0], cmap="gray")
    ax.set_title("Video 0, Frame 0")

    # --- Slider for Video ---
    ax_video = plt.axes([0.2, 0.2, 0.6, 0.03])
    slider_video = Slider(ax_video, "Video", 0, n_videos-1, valinit=0, valstep=1)

    # --- Slider for Frame ---
    ax_frame = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider_frame = Slider(ax_frame, "Frame", 0, n_frames-1, valinit=0, valstep=1)

    # Update function
    def update(val):
        vid_idx = int(slider_video.val)
        frame_idx = int(slider_frame.val)
        img.set_data(data[vid_idx, frame_idx])
        ax.set_title(f"Video {vid_idx}, Frame {frame_idx}")
        fig.canvas.draw_idle()

    slider_video.on_changed(update)
    slider_frame.on_changed(update)

    plt.show()

   
#%% visualize the data using frame viewer
#% Path to the nd2 file 
data_path = r'/Users/khadijehmasumnia/Codes/ML_Marathon/Dataset/240922_mattLines.nd2'
frame_viewer(data_path)

#%% visaulizing a sample single frame 

# plt.imshow(data[0, 0, :, :], cmap="gray")
# plt.show()   


#%% Save as lossless TIFF stack 

"""
Save ND2 video as TIFF stack and view TIFF stacks
"""

#%% Function: Save one ND2 video to TIFF stack
def save_video_as_tiff(nd2_path, video_index, tiff_path):
    with nd2.ND2File(nd2_path) as f:
        data = f.asarray().swapaxes(0,1)
        print("ND2 data shape:", data.shape)  
        # usually: (n_videos, n_frames, height, width)
        
        video = data[video_index]  # select video
        print(f"Saving Video {video_index}, shape: {video.shape} -> {tiff_path}")
        
        # Save as lossless TIFF stack
        tiff.imwrite(tiff_path, video, photometric="minisblack")

#%% Function: View TIFF stack interactively
import matplotlib
# Use TkAgg for scripts; remove or change for notebooks
matplotlib.use("TkAgg")  
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import tifffile as tiff
import numpy as np

def tiff_stack_viewer(tiff_path):
    """
    Interactive TIFF stack viewer with a slider for frames.
    Supports multi-frame TIFFs with any pixel depth.
    """
    # Load TIFF
    video = tiff.imread(tiff_path)  # shape: (n_frames, height, width)
    if video.ndim == 2:
        video = video[np.newaxis, :, :]  # convert single-frame to 3D

    n_frames, height, width = video.shape
    print(f"Video shape: {video.shape}, dtype: {video.dtype}")

    # Convert to float for safe display
    video_display = video.astype(np.float32)
    video_display /= video_display.max()  # normalize 0-1 for display

    # Create figure
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    img = ax.imshow(video_display[0], cmap="gray")
    ax.set_title("Frame 0")

    # Slider axis
    ax_frame = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_frame, "Frame", 0, n_frames-1, valinit=0, valstep=1)

    # Update function
    def update(val):
        frame_idx = int(slider.val)  # safely convert float to int
        img.set_data(video_display[frame_idx])
        ax.set_title(f"Frame {frame_idx}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show(block=True)

#%% Save first video as TIFF stack
save_tiff = r"/Users/khadijehmasumnia/Codes/ML_Marathon/nd2_save_tiff.tif"
save_video_as_tiff(data_path, video_index=0, tiff_path = save_tiff)

#%% View saved TIFF stack
tiff_file = r"/Users/khadijehmasumnia/Codes/ML_Marathon/nd2_save_tiff.tif"
tiff_stack_viewer(tiff_file)












