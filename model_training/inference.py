#!/usr/bin/env python3
"""
Author : Emmanuel Miguel Gonzalez
Date   : 2023-12-27
Purpose: Charcoal rot of sorghum (CRS) inference
"""

import argparse
import os
import sys
sys.path.append("..")
from models import *
from skimage.io import imread
import matplotlib.pyplot as plt

# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Charcoal rot of sorghum (CRS) inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('positional',
                        metavar='str',
                        help='A positional argument')

    parser.add_argument('-c',
                        '--checkpoint',
                        help='Path to model checkpoint directory',
                        metavar='str',
                        type=str,
                        required=True)

    parser.add_argument('-m',
                        '--model',
                        help='Name of model to use for inference',
                        metavar='str',
                        type=str,
                        choices=['DeepLabV3', 'EfficientNetB4', 'MobileNetV3Large', 'MobileNetV3SmallCustom',
                        'UNET', 'EfficientNetB3', 'FCN', 'MobileNetV3Small', 'ResNet'],
                        required=True)

    parser.add_argument('-v',
                        '--version',
                        help='Model version to use for inference',
                        metavar='str',
                        type=str,
                        choices=['version_0', 'version_1', 'version_2', 'version_3',
                        'version_4', 'version_5', 'version_6'],
                        default='version_0')

    parser.add_argument('-i',
                        '--image',
                        help='Path to image to run inference',
                        metavar='str',
                        type=str,
                        required=True)

    parser.add_argument('-o',
                        '--output_directory',
                        help='Output directory to save outputs',
                        metavar='str',
                        type=str,
                        default='crs_output')

    return parser.parse_args()


# --------------------------------------------------
def main():
    """Make a jazz noise here"""

    args = get_args()

    if not os.path.isdir(args.output_directory):
        os.makedirs(args.output_directory)
    
    base_path = args.model_checkpoint
    model_name = args.model
    version_no = args.version

    checkpoints_path = os.path.join(base_path,model_name,"lightning_logs", version_no, "checkpoints")
    checkpoint_name = os.listdir(checkpoints_path)[0]

    print(f"checkpoint name: {checkpoint_name}")
    model = eval(model_name).load_from_checkpoint(os.path.join(checkpoints_path, checkpoint_name))
    
    # Read the image
    image = imread(os.path.join(args.image))
    prediction = model.predict_single_image(image)

    # Create a subplot with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2)

    # Display the image in the first subplot
    axes[0].imshow(image)
    axes[0].set_title('Image')

    # Display the prediction in the second subplot
    axes[1].imshow(prediction, alpha=.5)
    axes[1].set_title('Prediction')

    # Save the plot
    plt.savefig(f'{args.output_directory}/output.png')

    # Show the plot
    plt.show()


# --------------------------------------------------
if __name__ == '__main__':
    main()
