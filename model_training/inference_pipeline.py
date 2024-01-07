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
import glob
import shutil
import time
import numpy as np
import pandas as pd


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
                        '--image_path',
                        help='Path to image',
                        metavar='str',
                        type=str,
                        required=True)
    
    return parser.parse_args()


# --------------------------------------------------
def get_model_cyverse_path(model_name):

    base_path = 'https://data.cyverse.org/dav-anon/iplant/projects/phytooracle/papers/CharcoalRotSorghum/model_checkpoints'

    model_path_dictionary = {
        'FCN': os.path.join(base_path, 'FCN/lightning_logs/version_0/checkpoints/epoch%3D41-step%3D247338.ckpt'), 
        'U-NET': os.path.join(base_path, 'UNET/lightning_logs/version_0/checkpoints/epoch%3D45-step%3D270894.ckpt'),
        'DeepLabV3': os.path.join(base_path, 'DeepLabV3/lightning_logs/version_6/checkpoints/epoch%3D44-step%3D83385.ckpt'), #'DeepLabV3/lightning_logs/version_0/checkpoints/epoch%3D45-step%3D270894.ckpt'),
        
        'EfficientNet-B4': os.path.join(base_path, 'EfficientNetB4/lightning_logs/version_0/checkpoints/epoch%3D2-step%3D17667.ckpt'),
        'MobileNetV3 large': os.path.join(base_path, 'MobileNetV3Large/lightning_logs/version_0/checkpoints/epoch%3D10-step%3D64779.ckpt'),
        'MobileNetV3 small custom': os.path.join(base_path, 'MobileNetV3SmallCustom/lightning_logs/version_0/checkpoints/epoch%3D6-step%3D41223.ckpt'),
        'EfficientNet-B3': os.path.join(base_path, 'EfficientNetB3/lightning_logs/version_0/checkpoints/epoch%3D9-step%3D58890.ckpt'),
        'MobileNetV3 small': os.path.join(base_path, 'MobileNetV3Small/lightning_logs/version_0/checkpoints/epoch%3D7-step%3D47112.ckpt'),
        'ResNet18': os.path.join(base_path, 'ResNet/lightning_logs/version_0/checkpoints/epoch%3D11-step%3D70668.ckpt')
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
    axes[0].set_title('Input Image')
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
def load_model(model_name, checkpoint_path):
    model_name = convert_names(model_name)
    return eval(model_name).load_from_checkpoint(checkpoint_path)


# --------------------------------------------------
def convert_names(model_name):
    model_name_dict = {
        'DeepLabV3': 'DeepLabV3',
        'FCN': 'FCN',
        'U-NET': 'UNET',
        'EfficientNet-B3': 'EfficientNetB3',
        'EfficientNet-B4': 'EfficientNetB4',
        'MobileNetV3 small': 'MobileNetV3Small',
        'MobileNetV3 small custom': 'MobileNetV3SmallCustom',
        'MobileNetV3 large': 'MobileNetV3Large',
        'ResNet18': 'ResNet'
    }

    return model_name_dict[model_name]


# --------------------------------------------------
def select_model():

    args = get_args()
    st.header("Select Model")
    
    # Select model
    model_name = st.selectbox(
        label="Select a Model", 
        options=["C: EfficientNet-B3", "C: EfficientNet-B4", "C: MobileNetV3 large", "C: MobileNetV3 small", "C: MobileNetV3 small custom", "C: ResNet18", "S: DeepLabV3", "S: FCN", "S: U-NET", ],
        index=None,
        placeholder="Select model...")

    model = None
    if model_name is not None:
        # st.sidebar.write('You selected:', model_name)
        # model_name = model_name.split(' ')[-1]
        model_name = ' '.join(model_name.split(' ')[1:])
        checkpoint_path = get_model_cyverse_path(model_name=model_name)
        model = load_model(model_name=model_name, checkpoint_path=checkpoint_path)

    return model, model_name


# --------------------------------------------------
def run_model(model, model_name, image_path):
    
    args = get_args()
    image = imread(image_path)

    if image is not None:
        # Run inference
        start_time = time.time()
        prediction = model.predict_single_image(image)
        end_time = time.time()
        execution_time = end_time - start_time
        return image, prediction, model_name, execution_time
    

# --------------------------------------------------
def model_results(image, prediction, model_name, execution_time):
    
    args = get_args()
    create_directory(directory_path=args.output_directory)
    
    if model_name in ['U-NET', 'FCN', 'DeepLabV3']:
        result_image_path = generate_plot(image=image, prediction=prediction, model=model_name)
        
        # Calculate the percentage of pixels with value 1 compared to those of value 0
        total_pixels = prediction.size
        ones = np.count_nonzero(prediction)
        zeros = total_pixels - ones
        percentage = (ones / total_pixels) * 100

        # Determine the presence of CRS
        presence = "Detected" if percentage > 0 else "Not Detected"
        # delete_directory(args.output_directory)

    else:
        presence = "Detected" if prediction==1 else "Not Detected"
        percentage = np.nan

    result = {
        'image_name': os.path.basename(args.image_path),
        'model_name': model_name,
        'execution_time': execution_time,
        'percentage_crs': percentage,
        'presence_crs': presence
    }
    
    return result


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

    result_dict = {}
    cnt = 0
    for model_name in ["C: EfficientNet-B3", "C: EfficientNet-B4", "C: MobileNetV3 large", "C: MobileNetV3 small", "C: MobileNetV3 small custom", "C: ResNet18", "S: DeepLabV3", "S: FCN", "S: U-NET", ]:
        cnt += 1
        model_name = ' '.join(model_name.split(' ')[1:])
        checkpoint_path = get_model_cyverse_path(model_name=model_name)
        model = load_model(model_name=model_name, checkpoint_path=checkpoint_path)
        if model is not None:
            image, prediction, model_name, execution_time = run_model(model=model, model_name=model_name, image_path=args.image_path)
            result = model_results(image=image, prediction=prediction, model_name=model_name, execution_time=execution_time)
            result_dict[cnt] = result
    
    result_df = pd.DataFrame.from_dict(result_dict, orient='index')
    result_df = result_df.fillna('NA')
    result_df.to_csv(os.path.join(args.output_directory, 'crs_results.csv'), index=False)


# --------------------------------------------------
if __name__ == '__main__':
    main()
