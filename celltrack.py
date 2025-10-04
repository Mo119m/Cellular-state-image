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

#%% loading and reading the data
# data_path = r'/Users/khadijehmasumnia/Codes/ML_Marathon/Dataset/240922_mattLines.nd2'

# with nd2.ND2File(data_path) as f:
#     print(f.shape)
#     print(f.size)
#     print(f.metadata)
    
#     data_orig = f.asarray()
#     data = data_orig.swapaxes(0, 1)

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


def tiff_stack_viewer(tiff_path):
    """
    Interactive TIFF stack viewer with a slider for frames.
    Supports multi-frame TIFFs with any pixel depth.
    """
    # Load TIFF
    video = tiff.imread(tiff_path)  # shape: (n_frames, height, width)
    print(video.shape)
    if video.ndim == 4:
        n_frames, height, width, channels = video.shape
    elif video.ndim == 2:
        video = video[np.newaxis, :, :]  # convert single-frame to 3D
    else:
        n_frames, height, width = video.shape
    
    #n_frames, height, width = video.shape
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


#%%
import xml.etree.ElementTree as ET
import pandas as pd

def save_cvat_csv(xml_file, save_path):
    def parse_cvat_xml(xml_file):
        """Parse CVAT XML annotation file into a DataFrame.
        
        Returns a DataFrame with columns:
        - frame: frame index
        - track_id: unique object ID
        - label: object class
        - group_id: optional group info
        - shape: type of shape (polygon, box, etc.)
        - attributes: dictionary of shape attributes
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        records = []
        for track in root.findall(".//track"):
            track_id = track.attrib["id"]
            label = track.attrib["label"]
            group_id = track.attrib.get("group_id", None)
            
            for shape in track:
                frame = int(shape.attrib["frame"])
                shape_type = shape.tag
                attrs = shape.attrib
                records.append({
                    "frame": frame,
                    "track_id": int(track_id),
                    "label": label,
                    "group_id": group_id,
                    "shape": shape_type,
                    "attributes": attrs
                })
        
        return pd.DataFrame(records)
    
    # save for easy exploration
    df = parse_cvat_xml(xml_file)
    df.to_csv(save_path, index=False)


#%% read annotations


import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from PIL import Image
import ast
from tqdm import tqdm


def annotate_tiff_stack(tiff_path, annotations_csv, output_tiff_path):
    # --- Load data ---
    frames = tifffile.imread(tiff_path)
    n_frames = frames.shape[0]
    df = pd.read_csv(annotations_csv)
    
    # Convert 'attributes' strings to actual dicts
    if isinstance(df.loc[0, "attributes"], str):
        df["attributes"] = df["attributes"].apply(ast.literal_eval)
    
    # --- Drawing function ---
    def draw_annotations_on_frame(frame_idx, frame):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        if frame.ndim == 2:
            ax.imshow(frame, cmap="gray")
        else:
            ax.imshow(frame)
    
        frame_df = df[df["frame"] == frame_idx]
    
        for _, row in frame_df.iterrows():
            shape_type = row["shape"].lower()
            attr = row["attributes"]
            label = row["label"]
    
            try:
                if shape_type == "ellipse":
                    cx = float(attr["cx"])
                    cy = float(attr["cy"])
                    rx = float(attr["rx"])
                    ry = float(attr["ry"])
                    ellipse = patches.Ellipse(
                        (cx, cy), 2 * rx, 2 * ry,
                        fill=False, edgecolor="red", linewidth=1
                    )
                    ax.add_patch(ellipse)
    
                elif shape_type == "rectangle":
                    x = float(attr["x"])
                    y = float(attr["y"])
                    w = float(attr["width"])
                    h = float(attr["height"])
                    rect = patches.Rectangle(
                        (x, y), w, h,
                        fill=False, edgecolor="blue", linewidth=1
                    )
                    ax.add_patch(rect)
    
                elif shape_type == "polygon":
                    points = attr.get("points")
                    if isinstance(points, str):
                        points = [tuple(map(float, p.split(","))) for p in points.split(";")]
                    polygon = patches.Polygon(points, closed=True, fill=False, edgecolor="lime", linewidth=1)
                    ax.add_patch(polygon)
    
                elif shape_type == "point":
                    cx = float(attr["cx"])
                    cy = float(attr["cy"])
                    ax.plot(cx, cy, 'yo', markersize=3)
    
                # add label text
                if shape_type in ["ellipse", "rectangle", "polygon", "point"]:
                    ax.text(float(attr.get("cx", x)), float(attr.get("cy", y)),
                            label, color="yellow", fontsize=6)
    
            except Exception as e:
                print(f"Warning on frame {frame_idx}, skipping shape {shape_type}: {e}")
    
        ax.axis("off")
    
        # Save figure to numpy array
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        return np.array(img)
    
    # --- Generate annotated frames ---
    annotated_frames = []
    
    print("Rendering annotated frames...")
    for i in tqdm(range(n_frames)):
        annotated_frame = draw_annotations_on_frame(i, frames[i])
        annotated_frames.append(annotated_frame)
    
    annotated_frames = np.array(annotated_frames)
    
    # --- Save as TIFF stack ---
    tifffile.imwrite(output_tiff_path, annotated_frames.astype(np.uint8))
    print(f"✅ Annotated TIFF saved to: {output_tiff_path}")











