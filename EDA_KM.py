#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 17:49:44 2025

@author: khadijehmasumnia
"""

import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt

#%% ========== Step 1: Parse XML ==========
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


#%% ========== Step 2: Load XML and inspect ==========
xml_file = "/Users/khadijehmasumnia/Codes/ML_Marathon/annotations.xml"   # <-- change this to your CVAT XML file path
df = parse_cvat_xml(xml_file)

print("First rows of parsed data:")
print(df.head())
print("\nSummary:")
print(df.info())

#%% ========== Step 3: EDA ==========
#% Group by frame and label, count unique cells (track_id)
cells_per_type_per_frame = (
    df.groupby(["frame", "label"])["track_id"].nunique().unstack(fill_value=0)
)

print(cells_per_type_per_frame.head())  # preview

# Plot
plt.figure(figsize=(12,6))
for label in cells_per_type_per_frame.columns:
    plt.plot(
        cells_per_type_per_frame.index, 
        cells_per_type_per_frame[label], 
        marker="o", 
        label=label
    )

plt.xlabel("Frame")
plt.ylabel("Number of Cells")
plt.title("Cell counts per frame by type")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xlim([0, 26])
plt.show()

#%%%%%%%%%%%%%%%%%  analysing frame intensities %%%%%%%%%%%%%%%%%%%%%%%%

import tifffile as tiff

vid_path = r'/Users/khadijehmasumnia/Codes/ML_Marathon/nd2_save_tiff.tif'

video = tiff.imread(vid_path)  # shape: (n_frames, height, width)
print(video.shape, video.dtype)

#%%
import cv2
import numpy as np
import pandas as pd

frame_stats = []

for i, frame in enumerate(video):
    mean_intensity = frame.mean()
    std_intensity = frame.std()
    
    # Variance of Laplacian -> focus measure
    focus_measure = cv2.Laplacian(frame, cv2.CV_64F).var()
    
    frame_stats.append({
        "frame": i,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "focus_measure": focus_measure
    })

stats_df = pd.DataFrame(frame_stats)
print(stats_df.head())

#%%

labels_per_frame = (
    df.groupby(["frame", "label"])["track_id"].nunique().unstack(fill_value=0)
)

# Merge with intensity stats
merged = stats_df.merge(labels_per_frame, left_on="frame", right_index=True, how="left")
print(merged.head())


#%%

# fig, ax1 = plt.subplots(figsize=(12,6))

# ax1.plot(merged["frame"], merged["focus_measure"], color="blue", label="Focus measure")
# ax1.set_ylabel("Focus measure", color="blue")

# ax2 = ax1.twinx()
# ax2.plot(merged["frame"], merged["biofilm"], color="red", marker="o", label="Biofilm count")
# ax2.set_ylabel("Biofilm count", color="red")

# plt.title("Biofilm count vs Focus over frames")
# plt.show()

#%%
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

scaler = MinMaxScaler()
metrics = merged[["mean_intensity", "std_intensity", "focus_measure"]]
normalized = pd.DataFrame(scaler.fit_transform(metrics), 
                          columns=metrics.columns)

plt.figure(figsize=(10, 5))
plt.plot(merged["frame"], normalized["mean_intensity"], "-o", label="Mean Intensity")
plt.plot(merged["frame"], normalized["std_intensity"], "-o", label="Std. Deviation")
plt.plot(merged["frame"], normalized["focus_measure"], "-o", label="Focus Measure")
plt.xlabel("Frame")
plt.ylabel("Normalized Value (0–1)")
plt.title("Normalized Frame-level Statistics")
plt.legend()
plt.xlim([0,26 ])
plt.grid(True)
plt.show()
