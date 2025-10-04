#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 12:35:55 2025

@author: khadijehmasumnia
"""

#%% Load Libraries
import celltrack

#%% DATA Path
# tiff_file = r"/Users/khadijehmasumnia/Codes/ML_Marathon/Dataset/Labeled_Data/LizaMutant38.tif" 
#tiff_file = r"/Users/khadijehmasumnia/Codes/ML_Marathon/Dataset/Labeled_Data/MattLines1.tif" 
tiff_file = r"/Users/khadijehmasumnia/Codes/ML_Marathon/Dataset/Labeled_Data/MattLines7.tif" 
#tiff_file = r"/Users/khadijehmasumnia/Codes/ML_Marathon/Dataset/Labeled_Data/MattLines27.tif" 

  
#%% visualize the data using frame viewer
# data_path = r'/Users/khadijehmasumnia/Codes/ML_Marathon/Dataset/240922_mattLines.nd2'
# celltrack.frame_viewer(data_path)

#%% Save first video as TIFF stack
# save_tiff = r"/Users/khadijehmasumnia/Codes/ML_Marathon/nd2_save_tiff.tif"
# save_video_as_tiff(data_path, video_index=0, tiff_path = save_tiff)

#%% View saved TIFF stack

#celltrack.tiff_stack_viewer(tiff_file)

#%%
xml_file = r"/Users/khadijehmasumnia/Codes/ML_Marathon/Dataset/Labeled_Data/MattLines7.xml" 
xml_save_path = r"/Users/khadijehmasumnia/Codes/ML_Marathon/Dataset/Labeled_Data/MattLines7.csv" 
celltrack.save_cvat_csv(xml_file, xml_save_path)



#%% --- Input paths ---

#tiff_path = r'/Users/khadijehmasumnia/Codes/ML_Marathon/Dataset/Labeled_Data/LizaMutant38.tif'
# annotations_csv = r'/Users/khadijehmasumnia/Codes/ML_Marathon/Dataset/Labeled_Data/LizaMutant38.csv'
#output_tiff_path = r'/Users/khadijehmasumnia/Codes/ML_Marathon/Dataset/Labeled_Data/annotated_LizaMutant38.tiff'

output_tiff_path = r'/Users/khadijehmasumnia/Codes/ML_Marathon/Dataset/Labeled_Data/annotated_MattLines7.tiff'

celltrack.annotate_tiff_stack(tiff_file, xml_save_path, output_tiff_path)

 #%% View saved annotated TIFF stack
annotated_tiff_file = r"/Users/khadijehmasumnia/Codes/ML_Marathon/Dataset/Labeled_Data/annotated_MattLines7.tiff" #"nd2_save_tiff.tif"
celltrack.tiff_stack_viewer(annotated_tiff_file)   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    