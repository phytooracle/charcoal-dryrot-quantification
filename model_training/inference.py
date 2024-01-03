#!/usr/bin/env python3
"""
Author : Emmanuel Miguel Gonzalez
Date   : 2023-12-27
Purpose: Charcoal rot of sorghum (CRS) inference
"""

import argparse
import os
import sys
from models import *
from skimage.io import imread
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from streamlit_image_select import image_select
import glob
import shutil
import time


# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Charcoal rot of sorghum (CRS) inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-o',
                        '--output_directory',
                        help='Output directory to save outputs',
                        metavar='str',
                        type=str,
                        default='crs_output')

    parser.add_argument('-i',
                        '--image_directory',
                        help='Directory containing images',
                        metavar='str',
                        type=str,
                        default='images/test_patches/')
    
    return parser.parse_args()


# --------------------------------------------------
def get_model_cyverse_path(model_name):

    base_path = 'https://data.cyverse.org/dav-anon/iplant/projects/phytooracle/papers/CharcoalRotSorghum/model_checkpoints'

    model_path_dictionary = {
        'S: FCN': os.path.join(base_path, 'FCN/lightning_logs/version_0/checkpoints/epoch%3D41-step%3D247338.ckpt'), 
        'S: UNET': os.path.join(base_path, 'UNET/lightning_logs/version_0/checkpoints/epoch%3D45-step%3D270894.ckpt'),
        'S: DeepLabV3': os.path.join(base_path, 'DeepLabV3/lightning_logs/version_6/checkpoints/epoch%3D44-step%3D83385.ckpt'), #'DeepLabV3/lightning_logs/version_0/checkpoints/epoch%3D45-step%3D270894.ckpt'),
        
        'C: EfficientNetB4': os.path.join(base_path, 'EfficientNetB4/lightning_logs/version_0/checkpoints/epoch%3D2-step%3D17667.ckpt'),
        'C: MobileNetV3Large': os.path.join(base_path, 'MobileNetV3Large/lightning_logs/version_0/checkpoints/epoch%3D10-step%3D64779.ckpt'),
        'C: MobileNetV3SmallCustom': os.path.join(base_path, 'MobileNetV3SmallCustom/lightning_logs/version_0/checkpoints/epoch%3D6-step%3D41223.ckpt'),
        'C: EfficientNetB3': os.path.join(base_path, 'EfficientNetB3/lightning_logs/version_0/checkpoints/epoch%3D9-step%3D58890.ckpt'),
        'C: MobileNetV3Small': os.path.join(base_path, 'MobileNetV3Small/lightning_logs/version_0/checkpoints/epoch%3D7-step%3D47112.ckpt'),
        'C: ResNet': os.path.join(base_path, 'ResNet/lightning_logs/version_0/checkpoints/epoch%3D11-step%3D70668.ckpt')
        }

    return model_path_dictionary[model_name]

    
# --------------------------------------------------
def generate_plot(image, prediction, model):

    args = get_args()
    create_directory(directory_path=args.output_directory)

    # Create a subplot with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2)

    # Display the image in the first subplot
    axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)  # Remove ticks and labels for axes[0]

    # Display the prediction in the second subplot
    axes[1].imshow(image)
    axes[1].imshow(prediction, alpha=.5)
    axes[1].set_title('Prediction')
    axes[1].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)  # Remove ticks and labels for axes[1]

    # Save the plot
    plt.savefig(f'{args.output_directory}/output_{model}.png', dpi=900, facecolor='white', edgecolor='white', bbox_inches='tight')

    # # Show the plot
    # plt.show()

    return f'{args.output_directory}/output_{model}.png'


# --------------------------------------------------
def documentation():
    st.markdown(
    """
    Charcoal rot of sorghum (CRS) is a disease caused by the fungal pathogen *Macrophomina phaseolina* 
    (Tassi) Goid. This fungal pathogen has a wide host range, infecting over 500 plant species in over 100 
    plant families. When *M. phaseolina* infects sorghum, it results in a variety of symptoms including root rot, 
    soft stalk, early lodging of plants, premature drying of stalk, reduced head size, and poor filling of 
    grain.

    This app allows you to run various classification and segmentation machine learning models that identify 
    and quanitfy CRS. Each trained model can be downloaded by clicking on the model name below:
    - Classification
        - [ResNet18](https://data.cyverse.org/dav-anon/iplant/projects/phytooracle/papers/CharcoalRotSorghum/model_checkpoints/ResNet/lightning_logs/version_0/checkpoints/epoch%3D11-step%3D70668.ckpt)
        - [MobileNetV3 small](https://data.cyverse.org/dav-anon/iplant/projects/phytooracle/papers/CharcoalRotSorghum/model_checkpoints/MobileNetV3Small/lightning_logs/version_0/checkpoints/epoch%3D7-step%3D47112.ckpt)
        - [MobileNetV3 small custom](https://data.cyverse.org/dav-anon/iplant/projects/phytooracle/papers/CharcoalRotSorghum/model_checkpoints/MobileNetV3SmallCustom/lightning_logs/version_0/checkpoints/epoch%3D6-step%3D41223.ckpt)
        - [MobileNetV3 large](https://data.cyverse.org/dav-anon/iplant/projects/phytooracle/papers/CharcoalRotSorghum/model_checkpoints/MobileNetV3Large/lightning_logs/version_0/checkpoints/epoch%3D10-step%3D64779.ckpt)
        - [EfficientNet-B3](https://data.cyverse.org/dav-anon/iplant/projects/phytooracle/papers/CharcoalRotSorghum/model_checkpoints/EfficientNetB3/lightning_logs/version_0/checkpoints/epoch%3D9-step%3D58890.ckpt)
        - [EfficientNet-B4](https://data.cyverse.org/dav-anon/iplant/projects/phytooracle/papers/CharcoalRotSorghum/model_checkpoints/EfficientNetB4/lightning_logs/version_0/checkpoints/epoch%3D2-step%3D17667.ckpt)
    - Segmentation
        - [U-NET](https://data.cyverse.org/dav-anon/iplant/projects/phytooracle/papers/CharcoalRotSorghum/model_checkpoints/UNET/lightning_logs/version_0/checkpoints/epoch%3D45-step%3D270894.ckpt)
        - [Fully Convolutional Network (FCN)](https://data.cyverse.org/dav-anon/iplant/projects/phytooracle/papers/CharcoalRotSorghum/model_checkpoints/FCN/lightning_logs/version_0/checkpoints/epoch%3D41-step%3D247338.ckpt)
        - [DeepLabV3](https://data.cyverse.org/dav-anon/iplant/projects/phytooracle/papers/CharcoalRotSorghum/model_checkpoints/DeepLabV3/lightning_logs/version_6/checkpoints/epoch%3D44-step%3D83385.ckpt)

    **To use this app: (*i*) select a model in the left sidebar, (*ii*) select or upload an image, (*iii*) scroll down to the model results.**
    """
    )


# --------------------------------------------------
@st.cache_resource
def load_model(model_name, checkpoint_path):
    return eval(model_name).load_from_checkpoint(checkpoint_path) #, # map_location='cpu'#'cuda:0')

# --------------------------------------------------
def input_upload_or_selection():
    
    args = get_args()

    st.header("Input Upload or Selection")
    st.markdown(
    """
    *To download the complete image patch test set and use it as input to the model, [click here](https://data.cyverse.org/dav-anon/iplant/projects/phytooracle/papers/CharcoalRotSorghum/images/test_images.zip).*
    """)
    # Load model
    model_name = st.sidebar.selectbox(
        "Select a Model", 
        ("S: DeepLabV3", "S: FCN", "S: UNET", "C: EfficientNetB3", "C: EfficientNetB4", "C: MobileNetV3Small", "C: MobileNetV3SmallCustom", "C: MobileNetV3Large", "C: ResNet"),
        placeholder="Select model...")
    st.sidebar.write('You selected:', model_name)
    checkpoint_path = get_model_cyverse_path(model_name=model_name)
    model = load_model(model_name=model_name, checkpoint_path=checkpoint_path)

    # Select an image from the library or upload an image
    # images = glob.glob(f'{args.image_directory}*.png')[:12]  # Replace with your actual image paths
    image_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    if image_file is not None:
        image = imread(image_file)
    else:
        selected_image = image_select(
            label="Select Image", 
            images=[
                f'{args.image_directory}IMG_0203-38.png',
                f'{args.image_directory}IMG_0547-32.png',
                f'{args.image_directory}IMG_0224-35.png',
                f'{args.image_directory}IMG_0762-21.png',
                f'{args.image_directory}IMG_0456-39.png',
                f'{args.image_directory}IMG_0683-26.png',
                f'{args.image_directory}IMG_0683-17.png',
                f'{args.image_directory}IMG_0683-22.png',
                f'{args.image_directory}IMG_0451-33.png',
                f'{args.image_directory}IMG_0449-32.png',
                f'{args.image_directory}IMG_0740-26.png',
                f'{args.image_directory}IMG_0754-40.png',
                f'{args.image_directory}IMG_0449-25.png',
                f'{args.image_directory}IMG_0449-33.png',
                f'{args.image_directory}IMG_0484-33.png',
                f'{args.image_directory}IMG_0740-38.png'
            ],
            captions=[
                'Plot stake',
                'Plot stake',
                'Control',
                'Control',
                'Control',
                'No CRS',
                'No CRS',
                'No CRS',
                'Major CRS',
                'Major CRS',
                'Major CRS',
                'Major CRS',
                'Major CRS',
                'Minor CRS',
                'Minor CRS',
                'Minor CRS'
            ],
            use_container_width=False)
        if selected_image is not None:
            image = imread(selected_image)

    if image is not None:
        # Run inference
        with st.spinner('Running model...'):
            start_time = time.time()
            prediction = model.predict_single_image(image)
            end_time = time.time()
            execution_time = end_time - start_time

        return image, prediction, model_name, execution_time


# --------------------------------------------------
# def model_results(image, prediction, model_name, execution_time):
#     st.header("Model Results")
#     st.success(f"{model_name} successfully made a prediction in {format(execution_time, '.2f')} seconds.")
#     args = get_args()
    
#     if model_name in ['UNET', 'FCN', 'DeepLabV3']:
#         result_image_path = generate_plot(image=image, prediction=prediction, model=model_name)
#         st.image(imread(result_image_path), caption=f'{model_name} Prediction', use_column_width=True)
#         delete_directory(args.output_directory)
#     else:
#         st.image(image, caption=f'{model_name} Prediction', use_column_width=True)
#         st.write('Classification: CRS positive' if prediction==1 else 'Classification: CRS negative')
import numpy as np

def model_results(image, prediction, model_name, execution_time):
    st.header("Model Results")
    st.success(f"{model_name} successfully made a prediction.")
    args = get_args()
    
    if model_name in ['UNET', 'FCN', 'DeepLabV3']:
        result_image_path = generate_plot(image=image, prediction=prediction, model=model_name)
        st.image(imread(result_image_path), caption=f'{model_name} Prediction', use_column_width=True)
        
        # Calculate the percentage of pixels with value 1 compared to those of value 0
        total_pixels = prediction.size
        ones = np.count_nonzero(prediction)
        zeros = total_pixels - ones
        percentage = (ones / total_pixels) * 100

        # Determine the presence of CRS
        presence = "True" if percentage > 0 else "False"
        
        # Display the metrics in a more understandable format
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Processing Time", value=f"{execution_time:.2f} s")
        col2.metric(label="Presence of CRS", value=presence)
        col3.metric(label="Percentage of Pixels with CRS", value=f"{percentage:.2f}%")
        
        delete_directory(args.output_directory)
    else:
        st.image(image, caption=f'{model_name} Prediction', use_column_width=True)
        presence = "True" if prediction==1 else "False"
        # st.write('Classification: CRS positive' if prediction==1 else 'Classification: CRS negative')
        col1, col2 = st.columns(2)
        col1.metric(label="Processing Time", value=f"{execution_time:.2f} s")
        col2.metric(label="Presence of CRS", value=presence)

# --------------------------------------------------
def delete_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    else:
        print("The directory does not exist")


# --------------------------------------------------
def create_directory(directory_path):
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)


# --------------------------------------------------
def main():
    """Make a jazz noise here"""
    args = get_args()
    st.title("Charcoal Rot of Sorghum Classification & Segmentation App")
    documentation()
    image, prediction, model_name, execution_time = input_upload_or_selection()
    model_results(image=image, prediction=prediction, model_name=model_name, execution_time=execution_time)


# --------------------------------------------------
if __name__ == '__main__':
    main()
