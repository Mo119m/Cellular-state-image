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









