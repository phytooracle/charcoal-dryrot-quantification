#!/usr/bin/env python3
"""
Author : Emmanuel Miguel Gonzalez
Date   : 2023-12-27
Purpose: Charcoal rot of sorghum (CRS) inference
"""

import argparse
import os
import sys
# sys.path.append("..")
from models import *
from skimage.io import imread
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from streamlit_image_select import image_select
import glob

# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Charcoal rot of sorghum (CRS) inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument('-c',
    #                     '--checkpoint',
    #                     help='Path to model checkpoint directory',
    #                     metavar='str',
    #                     type=str,
    #                     required=True)

    # parser.add_argument('-m',
    #                     '--model',
    #                     help='Name of model to use for inference',
    #                     metavar='str',
    #                     type=str,
    #                     choices=['DeepLabV3', 'EfficientNetB4', 'MobileNetV3Large', 'MobileNetV3SmallCustom',
    #                     'UNET', 'EfficientNetB3', 'FCN', 'MobileNetV3Small', 'ResNet'],
    #                     required=True)

    # parser.add_argument('-v',
    #                     '--version',
    #                     help='Model version to use for inference',
    #                     metavar='str',
    #                     type=str,
    #                     choices=['version_0', 'version_1', 'version_2', 'version_3',
    #                     'version_4', 'version_5', 'version_6'],
    #                     default='version_0')

    # parser.add_argument('-i',
    #                     '--image',
    #                     help='Path to image to run inference',
    #                     metavar='str',
    #                     type=str,
    #                     required=True)

    parser.add_argument('-o',
                        '--output_directory',
                        help='Output directory to save outputs',
                        metavar='str',
                        type=str,
                        default='crs_output')

    return parser.parse_args()


# --------------------------------------------------
def get_model_cyverse_path(model_name):

    base_path = 'https://data.cyverse.org/dav-anon/iplant/projects/phytooracle/papers/CharcoalRotSorghum/model_checkpoints'

    model_path_dictionary = {
        'FCN': os.path.join(base_path, 'FCN/lightning_logs/version_0/checkpoints/epoch%3D41-step%3D247338.ckpt'), 
        'UNET': os.path.join(base_path, 'UNET/lightning_logs/version_0/checkpoints/epoch%3D45-step%3D270894.ckpt'),
        'DeepLabV3': os.path.join(base_path, 'DeepLabV3/lightning_logs/version_6/checkpoints/epoch%3D44-step%3D83385.ckpt'), #'DeepLabV3/lightning_logs/version_0/checkpoints/epoch%3D45-step%3D270894.ckpt'),
        
        'EfficientNetB4': os.path.join(base_path, 'EfficientNetB4/lightning_logs/version_0/checkpoints/epoch%3D2-step%3D17667.ckpt'),
        'MobileNetV3Large': os.path.join(base_path, 'MobileNetV3Large/lightning_logs/version_0/checkpoints/epoch%3D10-step%3D64779.ckpt'),
        'MobileNetV3SmallCustom': os.path.join(base_path, 'MobileNetV3SmallCustom/lightning_logs/version_0/checkpoints/epoch%3D6-step%3D41223.ckpt'),
        'EfficientNetB3': os.path.join(base_path, 'EfficientNetB3/lightning_logs/version_0/checkpoints/epoch%3D9-step%3D58890.ckpt'),
        'MobileNetV3Small': os.path.join(base_path, 'MobileNetV3Small/lightning_logs/version_0/checkpoints/epoch%3D7-step%3D47112.ckpt'),
        'ResNet': os.path.join(base_path, 'ResNet/lightning_logs/version_0/checkpoints/epoch%3D11-step%3D70668.ckpt')
        }

    return model_path_dictionary[model_name]

    
# --------------------------------------------------
def generate_plot(image, prediction, model):

    args = get_args()

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

def documentation():
    st.markdown(
    """
    Charcoal rot of sorghum (CRS) is a disease caused by the fungal pathogen *Macrophomina phaseolina* 
    (Tassi) Goid. This fungal pathogen has a wide host range, infecting over 500 plant species in over 100 
    plant families. When sorghum is infected, it results in a variety of symptoms including root rot, 
    soft stalk, early lodging of plants, premature drying of stalk, reduced head size, and poor filling of 
    grain.

    This app allows you to run a variety of classification and segmentation models that identify and quanitfy 
    CRS. These models include:
    - Classification
        - ResNet18
        - MobileNetV3 small
        - MobileNetV3 small custom
        - MobileNetV3 large
        - EfficientNet-B3
        - EfficientNet-B4 (Koonce 2021a)
    - Segmentation
        - U-NET
        - Fully Convolutional Network (FCN)
        - DeepLabV3

    To use this app:
        1. Select or upload an image
        2. Scroll down to the model results 
    """
    )


# --------------------------------------------------
def input_upload_or_selection():
    st.header("Input Upload or Selection")
    # Load model
    model_name = st.sidebar.selectbox("Select Model", ("UNET", "FCN", "DeepLabV3",
    "EfficientNetB3", "EfficientNetB4", "MobileNetV3Small", "MobileNetV3SmallCustom", "MobileNetV3Large", "ResNet"))
    checkpoint_path = get_model_cyverse_path(model_name=model_name)
    model = eval(model_name).load_from_checkpoint(
        checkpoint_path,
        # map_location='cpu'#'cuda:0'
    )

    # Select an image from the library or upload an image
    images = glob.glob('images/test_patches/*.png')  # Replace with your actual image paths
    image_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    if image_file is not None:
        image = imread(image_file)
    else:
        selected_image = image_select("Select Image", images, use_container_width=True)
        if selected_image is not None:
            image = imread(selected_image)

    if image is not None:
        # Run inference
        prediction = model.predict_single_image(image)
        return image, prediction, model_name


# --------------------------------------------------
def model_results():
    st.header("Model Results")
    image, prediction, model_name = input_upload_or_selection()
    if model_name in ['UNET', 'FCN', 'DeepLabV3']:
        result_image_path = generate_plot(image=image, prediction=prediction, model=model_name)
        st.image(imread(result_image_path), caption=f'{model_name} Prediction', use_column_width=True)
    else:
        st.image(image)
        st.write('Classification: CRS positive' if prediction==1 else 'Classification: CRS negative')


# --------------------------------------------------
import streamlit as st

def main():
    """Make a jazz noise here"""
    args = get_args()

    if not os.path.isdir(args.output_directory):
        os.makedirs(args.output_directory)
    
    st.title("Charcoal Rot of Sorghum Classification & Segmentation App")
    documentation()
    input_upload_or_selection()
    model_results()


# --------------------------------------------------
if __name__ == '__main__':
    main()
