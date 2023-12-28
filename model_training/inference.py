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

# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Charcoal rot of sorghum (CRS) inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument('positional',
    #                     metavar='str',
    #                     help='A positional argument')

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
def main():
    """Make a jazz noise here"""

    args = get_args()

    if not os.path.isdir(args.output_directory):
        os.makedirs(args.output_directory)
    
    # base_path = args.checkpoint
    # model_name = args.model
    # version_no = args.version

    # checkpoints_path = os.path.join(base_path,model_name,"lightning_logs", version_no, "checkpoints")
    # checkpoint_name = os.listdir(checkpoints_path)[0]
    # model = eval(model_name).load_from_checkpoint(
    # os.path.join(checkpoints_path, checkpoint_name),
    # # map_location='cpu'#'cuda:0'
    # )
    # print(f"checkpoint name: {checkpoint_name}")
    checkpoint_path = get_model_cyverse_path(model_name=args.model)
    model = eval(args.model).load_from_checkpoint(
    checkpoint_path,
    # map_location='cpu'#'cuda:0'
    )
    
    # Read the image
    image = imread(os.path.join(args.image))

    # Run inference
    prediction = model.predict_single_image(image)
    
    if args.model in ['UNET', 'FCN', 'DeepLabV3']:

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
        plt.savefig(f'{args.output_directory}/output_{args.model}.png', dpi=900, facecolor='white', edgecolor='white', bbox_inches='tight')
    
    else:
        print('Classification: CRS positive' if prediction==1 else 'Classification: CRS negative')


# --------------------------------------------------
if __name__ == '__main__':
    main()
