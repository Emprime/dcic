import random
from os.path import join

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras import Model
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
import src.util.wide_resnet as wrn
from src.util.augmentation import  Augmenter
from src.util.const import get_all_dataset_infos
from src.util.json import DatasetDCICJson


def get_model(dataset_name, num_classes, weights=None, network_name="resnet50v2", dropout=0.5, input_upsampling = 0, overclustering_k = 0, get_gap_model = False):
    """
    get the model for the given network name
    :return:
    """

    #  parameters
    dataset_info = get_all_dataset_infos()[dataset_name]
    w_h = dataset_info.get_input_size() * (2 ** input_upsampling)


    # get backbone

    if input_upsampling == 0:

        input_tensor = None
    else:
        w_h = dataset_info.get_input_size()
        input = layers.Input((w_h,w_h,3))
        x = input
        for i in range(input_upsampling):
            x = layers.UpSampling2D((2, 2))(x)

        input_tensor = x

    if network_name == "resnet50v2" or network_name == "resnet50v2_large" or network_name == "resnet50v2_dc3" :
        backbone = ResNet50V2(include_top=False, weights=weights, input_shape=(w_h, w_h, 3), input_tensor=input_tensor, pooling='avg')
    elif network_name == "densenet121":
        backbone = DenseNet121(include_top=False, weights=weights, input_shape=(w_h, w_h, 3), input_tensor=input_tensor, pooling='avg')
    elif network_name == "incepresv2":
        backbone = InceptionResNetV2(include_top=False, weights=weights, input_shape=(w_h, w_h, 3), input_tensor=input_tensor, pooling='avg')
    elif network_name == "wideresnet16-8":
        backbone, x = wrn.create_wide_residual_network((w_h, w_h, 3), nb_classes=num_classes, N=2, k=8,
                                                       dropout=dropout, input_tensor=input_tensor)
    elif network_name == "wideresnet28-10":
        # For WRN-16-8 put N = 2, k = 8
        # For WRN-28-10 put N = 4, k = 10
        # For WRN-40-4 put N = 6, k = 4
        backbone, x = wrn.create_wide_residual_network((w_h, w_h, 3), nb_classes=num_classes, N=4, k=10, dropout=dropout, input_tensor=input_tensor)
    else:
        raise ValueError("%s is not an allowed network name" % network_name)

    # define output
    out = backbone.output

    if network_name == "wideresnet28-10":
        outputs = x
    elif overclustering_k > 0:
        # dc3 model
        x = layers.Dropout(dropout)(out)
        #output is num classes + overclustering + fuzziness
        outputs = layers.Dense(num_classes + num_classes*overclustering_k + 1)(x)
    elif network_name != "resnet50v2_large":
        x = layers.Dropout(dropout)(out)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
    else:
        # resnet60v2 large
        x = layers.Flatten()(out)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

    if input_upsampling == 0:
        model = Model(inputs=backbone.input, outputs=outputs, name="prediction_model")
        gap_model = Model(inputs=backbone.input, outputs=out, name="gap_model")
    else:
        model = Model(inputs=input, outputs=outputs, name="prediction_model")
        gap_model = Model(inputs=input, outputs=out, name="gap_model")


    if get_gap_model:
        return model, gap_model
    else:
        return model


def make_ds_from(dataset_info,paths,targets,augmentations, use_augmentations=False):
    """

    :param dataset_info:
    :param paths:
    :param targets:
    :param augmentations:
    :param use_augmentations:
    :return:
    """
    if augmentations is not None:
        augmenter = Augmenter(**augmentations)
    else:
        augmenter = Augmenter()

    dataset_root, dataset_name = dataset_info.raw_data_root_directory, dataset_info.name

    path_full = [join(dataset_root, path) for path in paths]

    list_ds = tf.data.Dataset.from_tensor_slices(path_full)
    if len(paths) > 0:
        images_ds = list_ds.map(get_parsing_function(dataset_name))
        images_ds = augmenter.apply_augmentation(images_ds, dataset_info.input_size,
                                                 use_augmenentation=use_augmentations)
    else:
        images_ds = list_ds  # no images loadable

    # print(soft_gt)

    labels_ds = tf.data.Dataset.from_tensor_slices(targets)

    return images_ds, labels_ds

def get_parsing_function(dataset_name):
    """
    get the parsing function of the input depending on the dataset_name

    :param dataset_name:
    :return:
    """

    # get input size depending on dataset name

    dataset_info = get_all_dataset_infos()[dataset_name]
    input_size = dataset_info.get_input_size()

    def parse_image(filename):
        """
        parse the image but apply no preprocessing
        :param filename:
        :return:
        """
        image = tf.io.read_file(filename)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, [input_size, input_size])
        return image

    return parse_image


