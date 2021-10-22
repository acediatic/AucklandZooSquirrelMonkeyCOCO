import cv2
from pathlib import Path
import argparse
import skimage
import mrcnn.model as modellib
from mrcnn.config import Config
import sys
import numpy as np
from termcolor import colored
from pathlib import Path
import os
import tensorflow as tf
from tensorflow import keras
import warnings
import pickle


class MonkeysConfig(Config):
    #################### BASE CONFIGURATION ####################
    NAME = "monkeys"

    # Train on 1 GPU
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 monkey
    DETECTION_MAX_INSTANCES = 1  # we're only looking for the most prominent individual in each image

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # RPN ANCHOR SCALES left as default (32, 64, 128, 256, 512), in line with the FaterRCNN paper
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    IMAGE_SHAPE = np.array([1024, 1024, 3])

    # (the number of batch iterations before a training epoch is considered finished). As we want to train on the full dataset, it's equal to num_samples/batch_size
    # STEPS_PER_EPOCH = 100

    # VALIDATION_STEPS is similiar to STEPS_PER_EPOCH


class InferenceConfig(MonkeysConfig):
    GPU_COUNT = 1
    # Batch size of 1 for inference
    IMAGES_PER_GPU = 1


def get_inference_model(inference_config, model_weights_path):
    print("Creating inference model")
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=os.getcwd())

    # Load trained weights
    print("Loading weights from ", model_weights_path)
    model.load_weights(model_weights_path, by_name=True)

    return model


def extract_faces(model_weights_path, image_path):
    inference_config = InferenceConfig()
    model_weights_path
    model = get_inference_model(inference_config, model_weights_path)

    # load the picture
    image = skimage.io.imread(image_path)

    print("Creating segmentation mask")
    # Run object detector
    results = model.detect([image], verbose=0)[0]
    # loop through the "results" to extract all regions of interest in the image

    print("Applying binary mask to image")
    try:
        # our best prediction appears to be our first
        mask_choice = 0

        # get the coordinates of the box containing the region of interest
        x = results["rois"][mask_choice][0]
        y = results["rois"][mask_choice][1]
        w = results["rois"][mask_choice][2]
        h = results["rois"][mask_choice][3]

        # transform every pixel that is not part of the region of interest to black
        mask = results["masks"][:, :, mask_choice].astype(np.uint8)
        contours, im2 = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        stencil = np.zeros(image.shape).astype(image.dtype)
        color = [255, 255, 255]
        cv2.fillPoly(stencil, contours, color)
        result = cv2.bitwise_and(image, stencil)

        # extract the box containing the region of interest and save the image
        result = result[x:w, y:h]

        return result

    except:
        print(colored(f"Unable to process'{image_path}'", 'red'))
        sys.exit(1)


def run_identification(vgg_model_path, knn_model_path, image):
    imsize = 150

    vgg_model = keras.models.load_model(vgg_model_path)

    vgg_output = vgg_model.predict(image.reshape(1, imsize, imsize, 3))

    knn_tl = pickle.loads(knn_model_path)
    prediction = knn_tl.predict(vgg_output)

    return prediction


def main(args):
    # Gets rid of a HOST of deprecation warnings for Matterport
    warnings.filterwarnings("ignore")
    # and Tensorflow
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    print(colored("Beginning image segmentation", "green"))
    segmented_image = extract_faces(args.model_path, args.image)
    print(colored("Completed image segmentation", "green"))

    print(colored("Beginning individual identification"), "green")
    prediction = run_identification(args.vgg_model_path, args.knn_model_path, segmented_image)

    print(colored(f"Predicted class is {prediction}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A pipeline for conspecific animal identification.')

    parser.add_argument("--image", help="The image of the individual to classify.", required=True)
    parser.add_argument("--mrcnn_model_path", help="The location of the mask rcnn model weights.", required=True)
    parser.add_argument("--vgg_model_path", help="The location of the vgg model.", required=True)
    parser.add_argument("--knn_model_path", help="The location of the knn model.", required=True)

    args = parser.parse_args()

    assert str(args.mrcnn_model_path).endswith(".h5"), "must be a weights file (.h5)"

    for arg in [args.image, args.mrcnn_model_path, args.vgg_model_path, args.knn_model_path]:
        if not os.path.isfile(arg):
            print(f"Error: {arg} is not a valid file path")
            sys.exit(1)

    main(args)
