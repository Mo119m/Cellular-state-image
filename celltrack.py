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

#%% Function: ND2 Frame_viewer

def frame_viewer(filepath):
    
    """ 
    this function reads the ND2 file directly and use a frameviewer 
    with a slider to slide through all videos and all frames 
    """
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
    """
    Save one ND2 video to TIFF stack
    
    Parameters
    ----------
    nd2_path : Path to the ND2 file
    video_index : video number which you like to save
    tiff_path : the path to save the TIFF stack

    """
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


#%% Function: Parse CVAT XML annotation file into a DataFrame.
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


#%% Function: annotate_tiff_stack
"""
read annotations and TIFF stack, 
annotate the TIFF stack and returns the annotated TIFF stack

"""

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
    #breakpoint()
    
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
            
            #breakpoint()
            
            # try:
            #     if label in ["yeast form" ,"planktonic", "psuedohyphae"]:
            #       edge_color="red" 
            #     elif label in ["single dispersed cell" ,"clump dispersed cell" ,"dispersed cell"]:
            #       edge_color="lime" 
    
            try:
                if label=="yeast form" :
                  edge_color="red" 
                elif label=="planktonic" :
                  edge_color="lime" 
                elif label=="psuedohyphae":
                  edge_color="magneta" 
                elif label=="single dispersed cell":
                  edge_color="blue"                   
                elif label=="clump dispersed cell":
                   edge_color="orange" 
                elif label=="dispersed cell":
                  edge_color="cyan"  
                elif label=="hyphae":
                  edge_color="yellow" 
                elif label=="biofilm":
                  edge_color="brown" 
                  
                

                # --- Ellipse ---
                if shape_type == "ellipse":
                    cx = float(attr["cx"])
                    cy = float(attr["cy"])
                    rx = float(attr["rx"])
                    ry = float(attr["ry"])
                    shape = patches.Ellipse(
                        (cx, cy), 2 * rx, 2 * ry,
                        fill=False, edgecolor=edge_color, linewidth=1
                    )
                    ax.add_patch(shape)
                    text_x, text_y = cx, cy
            
                # --- Polygon ---
                elif shape_type == "polygon":
                    points = attr.get("points")
                    if isinstance(points, str):
                        points = [tuple(map(float, p.split(","))) for p in points.split(";")]
                    shape = patches.Polygon(points, closed=True, fill=False, edgecolor=edge_color, linewidth=1)
                    ax.add_patch(shape)
                    # Use centroid for label placement
                    xs, ys = zip(*points)
                    text_x, text_y = sum(xs) / len(xs), sum(ys) / len(ys)
            
                # --- Polyline ---
                elif shape_type == "polyline":
                    points = attr.get("points")
                    if isinstance(points, str):
                        points = [tuple(map(float, p.split(","))) for p in points.split(";")]
                    xs, ys = zip(*points)
                    ax.plot(xs, ys, color=edge_color, linewidth=1)
                    text_x, text_y = xs[len(xs)//2], ys[len(ys)//2]
            
                else:
                    return  # skip unknown shapes
 
    
            except Exception as e:
                print(f"Warning on frame {frame_idx}, skipping shape {shape_type}: {e}")
    
        ax.axis("off")
    
        # Save figure to numpy array
        buf = io.BytesIO()
        #plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.savefig(buf, format="png", bbox_inches=None, pad_inches=0)

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


#%% Function: create_multiclass_masks: this function creates masks

import numpy as np
import pandas as pd
import tifffile
import ast
import cv2
from tqdm import tqdm

def create_multiclass_masks(tiff_path, annotations_csv, output_mask_path):
    """

    Parameters
    ----------
    tiff_path : path to original (non-annotated) video TIFF path
    annotations_csv : Path to CSV annotation file
    output_mask_path : output path to save the masks

    Returns
    -------
    None.

    """
    # --- Load data ---
    frames = tifffile.imread(tiff_path)
    n_frames, H, W = frames.shape
    df = pd.read_csv(annotations_csv)

    # Convert 'attributes' strings to dicts
    if isinstance(df.loc[0, "attributes"], str):
        df["attributes"] = df["attributes"].apply(ast.literal_eval)

    df["frame"] = df["frame"].astype(int)
    df["label"] = df["label"].str.strip().str.lower()

    # --- Define class IDs ---
    class_map = {
        "yeast form": 1,
        "planktonic": 2,
        "psuedohyphae": 3,
        "single dispersed cell": 4,
        "clump dispersed cell": 5,
        "dispersed cell": 6,
        "hyphae": 7,
        "biofilm": 8
    }

    # --- Create masks ---
    mask_stack = np.zeros((n_frames, H, W), dtype=np.uint8)

    print("Generating segmentation masks...")
    for i in tqdm(range(n_frames)):
        frame_df = df[df["frame"] == i]
        mask = np.zeros((H, W), dtype=np.uint8)

        for _, row in frame_df.iterrows():
            shape_type = row["shape"].lower()
            attr = row["attributes"]
            label = row["label"]
            class_id = class_map.get(label, 0)  # 0 = background

            try:
                # --- Ellipse ---
                if shape_type == "ellipse":
                    cx = int(float(attr["cx"]))
                    cy = int(float(attr["cy"]))
                    rx = int(float(attr["rx"]))
                    ry = int(float(attr["ry"]))
                    center = (cx, cy)
                    axes = (rx, ry)
                    cv2.ellipse(mask, center, axes, angle=0, startAngle=0, endAngle=360, color=class_id, thickness=-1)

                # --- Polygon ---
                elif shape_type == "polygon":
                    points = attr.get("points")
                    if isinstance(points, str):
                        points = [tuple(map(float, p.split(","))) for p in points.split(";")]
                    pts = np.array(points, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], color=class_id)

                # --- Polyline ---
                elif shape_type == "polyline":
                    points = attr.get("points")
                    if isinstance(points, str):
                        points = [tuple(map(float, p.split(","))) for p in points.split(";")]
                    pts = np.array(points, dtype=np.int32)
                    cv2.polylines(mask, [pts], isClosed=False, color=class_id, thickness=2)

            except Exception as e:
                print(f"⚠️  Warning: skipping shape '{shape_type}' in frame {i} — {e}")

        mask_stack[i] = mask

    # --- Save stack ---
    tifffile.imwrite(output_mask_path, mask_stack)
    print(f"✅ Multi-class mask stack saved to: {output_mask_path}")



#%%

import numpy as np
import os
from skimage.measure import regionprops, label
from skimage import io


def crop_and_pad_objects(image, mask, save_dir='crops'):
    """
    Crop each labeled object in the mask and pad all to the largest object size.
    Saves crops grouped by class label.
    
    image: 2D or 3D numpy array (H, W) or (H, W, C)
    mask: 2D labeled mask (same H, W)
    save_dir: output directory for saving crops
    """

    os.makedirs(save_dir, exist_ok=True)

    all_crops = []
    all_masks = []
    crop_shapes = []
    crop_info = []  # store (class, index)

    # Loop through each nonzero class label
    for cls in np.unique(mask):
        if cls == 0:
            continue

        class_mask = (mask == cls)
        labeled_class = label(class_mask)
        props = regionprops(labeled_class)

        for i, region in enumerate(props):
            minr, minc, maxr, maxc = region.bbox
            img_crop = image[minr:maxr, minc:maxc]
            mask_crop = mask[minr:maxr, minc:maxc]

            all_crops.append(img_crop)
            all_masks.append(mask_crop)
            crop_shapes.append(img_crop.shape[:2])
            crop_info.append((cls, i))

    # --- Find max crop size ---
    max_h = max(h for h, w in crop_shapes)
    max_w = max(w for h, w in crop_shapes)
    max_side = max(max_h, max_w)
    print(f"Maximum object size: {max_side} x {max_side}")

    # --- Pad all crops ---
    for (img, msk), (cls, i) in zip(zip(all_crops, all_masks), crop_info):
        h, w = img.shape[:2]
        pad_h = max_side - h
        pad_w = max_side - w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Pad image and mask symmetrically
        if img.ndim == 2:  # grayscale
            img_padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
        else:  # RGB
            img_padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')

        mask_padded = np.pad(msk, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

        # Save in class-specific folder
        class_dir = os.path.join(save_dir, f"class_{cls}")
        os.makedirs(class_dir, exist_ok=True)
        io.imsave(os.path.join(class_dir, f"crop_{i}_image.png"), img_padded)
        io.imsave(os.path.join(class_dir, f"crop_{i}_mask.png"), mask_padded)

    print(f"✅ Done! Crops saved in '{save_dir}' grouped by class labels.")
#%%


import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
import matplotlib.patches as patches

def show_crop_boxes(image, mask, class_map=None):
    """
    Displays the original image and mask with bounding boxes drawn
    around each labeled object (based on the mask).
    
    image: 2D or 3D numpy array
    mask: 2D labeled mask (same height/width)
    class_map: optional dict mapping class names -> label numbers
    """

    # --- Set up figure ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    ax_img, ax_mask = axes
    ax_img.imshow(image, cmap='gray')
    ax_img.set_title("Original Image with Crop Boxes")
    ax_mask.imshow(mask, cmap='nipy_spectral')
    ax_mask.set_title("Mask with Crop Boxes")

    # --- Colors ---
    cmap = plt.get_cmap("tab10")
    colors = {cls: cmap(i % 10) for i, cls in enumerate(np.unique(mask)) if cls != 0}

    # --- Draw boxes ---
    for cls in np.unique(mask):
        if cls == 0:
            continue
        class_mask = (mask == cls)
        labeled_class = label(class_mask)
        props = regionprops(labeled_class)

        for region in props:
            minr, minc, maxr, maxc = region.bbox
            color = colors[cls]

            rect = patches.Rectangle(
                (minc, minr), maxc - minc, maxr - minr,
                linewidth=2, edgecolor=color, facecolor='none'
            )

            # Draw on both image and mask
            ax_img.add_patch(rect)
            ax_mask.add_patch(patches.Rectangle(
                (minc, minr), maxc - minc, maxr - minr,
                linewidth=2, edgecolor=color, facecolor='none'
            ))

            # Label the class
            label_text = None
            if class_map:
                label_text = [k for k, v in class_map.items() if v == cls]
                label_text = label_text[0] if label_text else f"Class {cls}"
            else:
                label_text = f"Class {cls}"

            ax_img.text(minc, minr - 3, label_text, color=color, fontsize=8, weight='bold')
            ax_mask.text(minc, minr - 3, label_text, color=color, fontsize=8, weight='bold')

    plt.tight_layout()
    plt.show()

