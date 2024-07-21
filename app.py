import streamlit as st
import numpy as np
from PIL import Image
import cv2
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from scipy.ndimage import rotate
import math
import matplotlib.pyplot as plt
from helpers import (get_bounding_box_of_large_region, get_model_predictions, upsample_logits, get_predicted_segmentation, label_to_color, extract_roi, get_shirt_bounding_box, calculate_angle_between_peaks, get_blend_ratio, overlay_logo, get_combined_image,process_shirt_files)
import tempfile
import os
# Load the pre-trained models
processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer-b2-fashion")
model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer-b2-fashion")

# Streamlit app
st.set_page_config(layout="wide")

# Initialize session state for button presses and offsets
if "x_offset" not in st.session_state:
    st.session_state.x_offset = 0
if "y_offset" not in st.session_state:
    st.session_state.y_offset = 0
if "threshold_val" not in st.session_state:
    st.session_state.threshold_val = 0.3
if "intensity_factor" not in st.session_state:
    st.session_state.intensity_factor = 0.5
if "user_angle" not in st.session_state:
    st.session_state.user_angle = 0.0  # Initialize as float
if "initial_angle_set" not in st.session_state:
    st.session_state.initial_angle_set = False
if "initial_angle" not in st.session_state:
    st.session_state.initial_angle = 0.0  # Initialize as float
if "shirt_files" not in st.session_state:
    st.session_state.shirt_files = []
if "logo_files" not in st.session_state:
    st.session_state.logo_files = []
if "apply_clicked" not in st.session_state:
    st.session_state.apply_clicked = False
if "color_clicked" not in st.session_state:
    st.session_state.color_clicked = False
if "generate_clicked" not in st.session_state:
    st.session_state.generate_clicked = False
if "upload_counter" not in st.session_state:
    st.session_state.upload_counter = 0  # Counter for tracking uploads
if "angle_slider_changed" not in st.session_state:
    st.session_state.angle_slider_changed = False  # Track if angle slider is changed
if "acv_file" not in st.session_state:
    st.session_state.acv_file = None
# Initialize session state for RGB values
if "r_value" not in st.session_state:
    st.session_state.r_value = 1.0
if "g_value" not in st.session_state:
    st.session_state.g_value = 1.0
if "b_value" not in st.session_state:
    st.session_state.b_value = 1.0

# Initialize session state for shadows, highlights, and mid tones
if "shadw_value" not in st.session_state:
    st.session_state.shadw_value = 1.0
if "highl_value" not in st.session_state:
    st.session_state.highl_value = 1.0
if "midt_value" not in st.session_state:
    st.session_state.midt_value = 1.0

if "images_upload" not in st.session_state:
    st.session_state.images_upload = False
if "acv_upload" not in st.session_state:
    st.session_state.acv_upload = False
if "show_sliders" not in st.session_state:
    st.session_state.show_sliders = False

if 'acv_file_path' not in st.session_state:
    st.session_state.acv_file_path = None



# Function to handle file upload within a form
def upload_files():
    with st.form(key='file_upload_form'):
        shirt_files = st.file_uploader("Upload Shirt Images", accept_multiple_files=True)
        logo_files = st.file_uploader("Upload Logo Images", accept_multiple_files=True)
        #acv_file = st.file_uploader("Upload ACV File", type=['acv'])
        upload_button = st.form_submit_button(label='Upload')
        if upload_button:
            if shirt_files:
                st.session_state.shirt_files = [Image.open(file).convert("RGB") for file in shirt_files]
            if logo_files:
                st.session_state.logo_files = [Image.open(file).convert("RGBA") for file in logo_files]
            # if acv_file:
            #     st.session_state.acv_file = acv_file.read()
            
            st.session_state.upload_counter += 1
            st.session_state.initial_angle_set = False  # Reset initial angle set flag
            st.session_state.angle_slider_changed = False  # Reset slider changed flag
            st.session_state.user_angle = 0.0  # Reset slider to 0.0
            st.session_state.x_offset = 0
            st.session_state.y_offset = 0
            st.session_state.threshold_val = 0.5
            st.session_state.intensity_factor = 0.15
            st.session_state.r_value = 1
            st.session_state.g_value = 1
            st.session_state.b_value = 1
            st.session_state.images_upload = True

# Define the hard-coded path for storing the ACV file
hard_coded_path =  r"C:\AI graphic apparel streamlite\files"

# Make sure the directory exists
os.makedirs(hard_coded_path, exist_ok=True)

def adjust_rgb():
    uploaded_file = st.file_uploader("Upload ACV File", type=['acv'])

    if uploaded_file:
        acv_file_path = os.path.join(hard_coded_path, uploaded_file.name)
        with open(acv_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.acv_file_path = acv_file_path
        st.session_state.acv_upload = True
        st.write("ACV file uploaded successfully!")

    else:
        print("acv do not uploaded")
        st.session_state.acv_upload = False
        st.session_state.show_sliders = False

    # Show the Align Color button
    if st.button('Align Color'):
        if st.session_state.acv_upload:
            st.session_state.show_sliders = True
    
    # Show RGB sliders and Apply Color button only if Align Color is clicked and ACV file is uploaded
    if st.session_state.show_sliders:
        st.session_state.r_value = st.slider("R Value", 0.0, 2.0, float(st.session_state.r_value), step=0.1)
        st.session_state.g_value = st.slider("G Value", 0.0, 2.0, float(st.session_state.g_value), step=0.1)
        st.session_state.b_value = st.slider("B Value", 0.0, 2.0, float(st.session_state.b_value), step=0.1)

        st.session_state.shadw_value = st.slider("shadow Value", 0.0, 10.0, float(st.session_state.shadw_value), step=0.1)
        st.session_state.highl_value = st.slider("highlight Value", 0.0, 10.0, float(st.session_state.highl_value), step=0.1)
        st.session_state.midt_value = st.slider("mid tones Value", 0.0, 10.0, float(st.session_state.midt_value), step=0.1)

        # Add color buttons
        color_buttons = {
            "Red": (0.6, 0.1, 0.1),
            "Green": (0.1, 0.6, 0.1),
            "Blue": (0.1, 0.1, 0.6),
            "Yellow": (0.6, 0.6, 0.1),
            "Cyan": (0.1, 0.6, 0.6),
            "Magenta": (0.6, 0.1, 0.6),
            "White": (2,2,2),
            "Black": (0.1, 0.1, 0.1),
            "Gray": (0.3, 0.3, 0.3),
            "Orange": (0.6, 0.3, 0.1)
        }
        
        cols = st.columns(3)
        idx = 0
        for color, rgb in color_buttons.items():
            if cols[idx].button(color, key=color):
                st.session_state.r_value, st.session_state.g_value, st.session_state.b_value = rgb
                print("rgb : ",rgb)
            idx = (idx + 1) % 3

        if st.button('Apply Color'):
            st.session_state.color_clicked = True
            

# Function to handle slider adjustments within a form
def adjust_sliders():
    with st.form(key='slider_form'):
        st.session_state.threshold_val = st.slider("Threshold Value", 0.0, 1.0, st.session_state.threshold_val)
        st.session_state.intensity_factor = st.slider("Intensity Factor", 0.0, 1.0, st.session_state.intensity_factor)
        user_angle = st.slider("Adjust Logo Angle", -50.0, 50.0, st.session_state.user_angle, step=0.1)
        st.session_state.x_offset = st.slider("X Offset", -30, 30, st.session_state.x_offset)
        st.session_state.y_offset = st.slider("Y Offset", -30, 30, st.session_state.y_offset)
        
        apply_button = st.form_submit_button(label='Apply')
        generate_button = st.form_submit_button(label='Generate')
        if apply_button:
            st.session_state.angle_slider_changed = user_angle != st.session_state.user_angle
            st.session_state.user_angle = user_angle  # Update user angle
            st.session_state.apply_clicked = True
        if generate_button:
            st.session_state.generate_clicked = True
def apply(combined_images):

    #shirt_images = st.session_state.shirt_files
    shirt_images=combined_images
    logo_images = st.session_state.logo_files

    image = shirt_images[0]
    logo_image = logo_images[0]

    x, y, w, h, image, pred_seg, cleaned_segment = get_shirt_bounding_box(image, processor, model)

    if x is not None:
            
        color_seg = label_to_color(pred_seg)
        a, b, c, d, seg_image_path = get_bounding_box_of_large_region(color_seg)

        if not st.session_state.initial_angle_set:
            st.session_state.initial_angle = calculate_angle_between_peaks(seg_image_path, x, y, w, h)
            st.session_state.user_angle = st.session_state.initial_angle
            st.session_state.initial_angle_set = True

        # Use initial angle for the first application, then use user-adjusted angle
        #angle_to_use = st.session_state.initial_angle if not st.session_state.initial_angle_set else st.session_state.user_angle

        print("in apply (angle) : ", st.session_state.user_angle)
        warped = overlay_logo(image, logo_image, a, b, c, d, -st.session_state.user_angle, st.session_state.threshold_val, st.session_state.intensity_factor, st.session_state.x_offset, st.session_state.y_offset)
        st.image(warped, use_column_width=True)
        st.session_state.apply_clicked = False


def Generate(combined_images):

    print("R value is : ",st.session_state.r_value)
    print("G value is : ",st.session_state.g_value)
    print("B value is : ",st.session_state.b_value)
    #shirt_images = st.session_state.shirt_files
    shirt_images=combined_images
    logo_images = st.session_state.logo_files

    combinations = []

    # Iterate over shirt images
    for shirt_image in shirt_images:
        x, y, w, h, image, pred_seg, cleaned_segment = get_shirt_bounding_box(shirt_image, processor, model)
        if x is not None:
            color_seg = label_to_color(pred_seg)
            a, b, c, d, seg_image_path = get_bounding_box_of_large_region(color_seg)
                
            if not st.session_state.initial_angle_set:
                st.session_state.initial_angle = calculate_angle_between_peaks(seg_image_path, x, y, w, h)
                st.session_state.user_angle = st.session_state.initial_angle
                st.session_state.initial_angle_set = True

                # Calculate and store the initial angle
                #angle_to_use = st.session_state.user_angle

                # Save all combinations in a list
            for logo_image in logo_images:
                print("slider status : ", st.session_state.angle_slider_changed)
                print("angle to use : ", st.session_state.user_angle)
                warped = overlay_logo(image, logo_image, a, b, c, d, -st.session_state.user_angle, st.session_state.threshold_val, st.session_state.intensity_factor, st.session_state.x_offset, st.session_state.y_offset)
                x, y, w, h, image, pred_seg, cleaned_segment = get_shirt_bounding_box(shirt_image, processor, model)

                combinations.append(warped)
        st.session_state.initial_angle_set = False

    # Display combinations in rows with a maximum of 3 images per row
    images_per_row = 3
    for i in range(0, len(combinations), images_per_row):
        row_images = combinations[i:i + images_per_row]
        cols = st.columns(len(row_images))
        for col, img in zip(cols, row_images):
            col.image(img, use_column_width=True)

        # Reset apply button state after processing
    
# Create two columns with a 30/70 ratio
col1, col2 = st.columns([0.3, 0.7])

with col1:
    # Title
    st.markdown("<h1 style='font-size: 40px;'>AI Graphic Apparel</h1>", unsafe_allow_html=True)
    
    # Image upload sections
    upload_files()

    if st.session_state.shirt_files and st.session_state.logo_files:
        # Display sliders for adjustments if images are uploaded
        st.subheader("Adjustments")
        adjust_sliders()
        if st.session_state.images_upload:
            adjust_rgb()
        

with col2:

    if st.session_state.color_clicked:
        print("yes")

        combined_images = process_shirt_files(
            st.session_state.shirt_files, 
            st.session_state.acv_file_path, 
            st.session_state.r_value, 
            st.session_state.g_value, 
            st.session_state.b_value,

            st.session_state.shadw_value, 
            st.session_state.midt_value, 
            st.session_state.highl_value


        )
        #st.session_state.shirt_files = combined_images

        # for img in combined_images:
        #     st.image(img, use_column_width=True)
                
        apply(combined_images)
        st.session_state.color_clicked = False


    if st.session_state.apply_clicked:
        #print("inside apply clicked ! ")
        apply(st.session_state.shirt_files)

    if st.session_state.generate_clicked:
        print("inside generate clicked ! ")
        
        Generate(st.session_state.shirt_files)
        st.session_state.generate_clicked = False

    else:
        st.write("Please upload shirt images, logo images, and ACV file.")
