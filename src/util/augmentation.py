import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import imagenet_utils
#from imgaug import augmenters as iaa


# content motivated and partially copied from https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/

def plot_images(dataset, n_images, samples_per_image):
    output = np.zeros((32 * n_images, 32 * samples_per_image, 3))

    row = 0
    for images in dataset.repeat(samples_per_image).batch(n_images):
        output[:, row*32:(row+1)*32] = np.vstack(images.numpy())
        row += 1

    plt.figure()
    plt.imshow(output)
    plt.show()


class Augmenter():

    def __init__(self, prob_flip=0, prob_color=0, prob_rotate=0, prob_zoom=0, use_imgaug=False):
        self.prob_flip = prob_flip
        self.prob_color = prob_color
        self.prob_rotate = prob_rotate
        self.prob_zoom = prob_zoom
        self.use_imgaug = use_imgaug

        # set seed
        tf.random.set_seed(424242)



    def flip(self, x: tf.Tensor) -> tf.Tensor:
        """Flip augmentation

        Args:
            x: Image to flip

        Returns:
            Augmented image
        """

        def random_flip(x):
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_flip_up_down(x)
            return x

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        return tf.cond(choice > self.prob_flip, lambda: x, lambda: random_flip(x))


    def color(self, x: tf.Tensor) -> tf.Tensor:
        """Color augmentation

        Args:
            x: Image

        Returns:
            Augmented image
        """
        def random_color(x):
            x = tf.image.random_hue(x, 0.08)
            x = tf.image.random_saturation(x, 0.6, 1.6)
            x = tf.image.random_brightness(x, 0.2)
            x = tf.image.random_contrast(x, 0.5, 1.5)
            return x
        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        return tf.cond(choice > self.prob_color, lambda: x, lambda: random_color(x))

    def rotate(self, x: tf.Tensor) -> tf.Tensor:
        """Rotation augmentation

        Args:
            x: Image

        Returns:
            Augmented image
        """

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        return tf.cond(choice > self.prob_rotate, lambda: x, lambda: tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)))

    def zoom_size(self,w_h):
        def zoom(x: tf.Tensor) -> tf.Tensor:
            """Zoom augmentation

            Args:
                x: Image

            Returns:
                Augmented image
            """

            # Generate 20 crop settings, ranging from a 1% to 20% crop.
            scales = list(np.arange(0.8, 1.0, 0.01))
            boxes = np.zeros((len(scales), 4))

            for i, scale in enumerate(scales):
                x1 = y1 = 0.5 - (0.5 * scale)
                x2 = y2 = 0.5 + (0.5 * scale)
                boxes[i] = [x1, y1, x2, y2]

            def random_crop(img):
                # Create different crops for an image
                crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(w_h, w_h))
                # Return a random crop
                return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


            choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

            # Only apply cropping 50% of the time
            return tf.cond(choice > self.prob_zoom, lambda: x, lambda: random_crop(x))
        return zoom

    def imgaug(self, images):
        # ImgAug might not be combatible anymore with framework
        from imgaug import augmenters as iaa 
        augmenter = iaa.Sequential([
            # iaa.CropToFixedSize(width=target_size[0], height=target_size[1], position="uniform"),
            iaa.Fliplr(0.5),
            iaa.MultiplyHue(mul=(1 - 0.125, 1.125)),
            iaa.MultiplySaturation(mul=(0.6, 1.4)),
            iaa.Multiply(mul=(0.6, 1.4)),
            iaa.GammaContrast(gamma=(0.6, 1.4)),
            iaa.Cutout(nb_iterations=1, fill_mode="constant", cval=0, size=(0.2, 0.7)),
            iaa.Affine(
                rotate=(-20, 20),  # in degrees
                shear=(-10, 10),  # in degreees
                order=1,  # use bilinear interpolation (fast)
                cval=0,  # if mode is constant, use cval of 0
                mode='constant'  # constant mode
            )
        ])

        img_dtype = images.dtype
        img_shape = tf.shape(images)
        images = tf.dtypes.cast(images, tf.uint8)
        # print(tf.shape(images))
        images = tf.numpy_function(augmenter.augment_image,
                                   [images],
                                   tf.uint8)
        images = tf.dtypes.cast(images, img_dtype)
        images = tf.reshape(images, shape=img_shape)

        return images

    def apply_augmentation(self, dataset, w_h, use_augmenentation=False):
        """
        Apply a variety of augmentations to the given tf dataset
        :param dataset:
        :return:
        """

        # Add augmentations
        augmentations = [self.flip, self.color, self.zoom_size(w_h), self.rotate]

        # apply imgaug
        if use_augmenentation and self.use_imgaug:
            # imgaug expects 0 ... 255
            dataset = dataset.map(lambda x: self.imgaug(x), num_parallel_calls=tf.data.AUTOTUNE)

        # apply preprocessing
        dataset = dataset.map(lambda x: imagenet_utils.preprocess_input(x, mode='tf'),
                              num_parallel_calls=tf.data.AUTOTUNE)

        if use_augmenentation:
            for f in augmentations:
                # apply each augmenation only to 25% of the data independialy
                # dataset = dataset.map(lambda x: tf.cond(tf.random.uniform([], 0, 1) > 0.75, lambda: f(x), lambda: x), num_parallel_calls=tf.data.AUTOTUNE)
                dataset = dataset.map(f, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.map(lambda x: tf.clip_by_value(x, -1, 1), num_parallel_calls=tf.data.AUTOTUNE)

        return dataset


# visualize
def visualize(dataset):

    plot_images(dataset, n_images=8, samples_per_image=10)









