import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np


def preprocess_image(Image_path: str) -> tf.Tensor:
    """_summary_
        preprocess the image to be ready for the model
    Args:
        Image_path (str): the path to the image

    Returns:
        _type_: a tensor of the image
    """

    # * read the image
    image = tf.io.read_file(Image_path)
    # * decode the image
    image = tf.image.decode_image(image)

    # * shape wihtout the last dimension - channels
    img_shape = image.shape[:-1]

    # * the size of the image should be divisible by 4
    image_adjusted_size = (tf.convert_to_tensor(img_shape) // 4) * 4
    # * crop the image to the new size
    image_adjusted = tf.image.crop_to_bounding_box(
        image=image,
        offset_height=0,
        offset_width=0,
        target_height=image_adjusted_size[0],
        target_width=image_adjusted_size[1],
    )

    # * convert the image to float32
    image_adjusted_casted = tf.cast(image_adjusted, tf.float32)
    return image_adjusted_casted


## Save the img
def save_image(
    image: Image,
    filename: str,
):
    # * if the image is not an image, convert it to PIL image and save it with pillow
    if not isinstance(image, Image.Image):
        # * clip the image to be between 0 and 255
        image = tf.clip_by_value(image, 0, 255)
        # * convert the image to uint8
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
        image.save(fp=filename + ".jpg", format="JPEG")


## plotting the images
def plot_image(
    image: tf.Tensor, title: str = "", streamlit: bool = False
) -> plt.figure or None:
    fig = plt.figure()
    image = np.asarray(image)
    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, tf.uint8).numpy()
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    if streamlit == True:
        return fig


def downscale_image(image: tf.Tensor) -> tf.Tensor:

    image_size = []
    if len(image.shape) == 3:
        image_size = [image.shape[0], image.shape[1]]

    else:
        raise ValueError("Make that it's a single image in Png or Jpg format")

    # * convert the image to uint8 + squeeze the image
    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, tf.uint8)
    image = tf.squeeze(image)

    # * resize the image to be 1/4 of the original size with BICUBIC interpolation
    # * the model was trained on images that are 1/4 of the original size downsampled with BICUBIC interpolation
    image = image.numpy()  # * convert the image to numpy
    image = Image.fromarray(image)  # * convert the image to PIL image
    image = image.resize(
        (image_size[0] // 4, image_size[1] // 4), Image.Resampling.BICUBIC
    )  # * resize the image
    image = np.asarray(image)  # * convert the image back to numpy

    # image = tf.expand_dims(image, axis=0) #* expand the image to be 4D (batch, height, width, channels
    # image = tf.cast(image, tf.float32) #* convert the image to float32

    return image


def prepare_for_model(image: tf.Tensor) -> tf.Tensor:
    """_summary_
        preprocess the image to be ready for the model
    Args:
        Image_path (str): the path to the image

    Returns:
        _type_: a tensor of the image
    """
    image = tf.expand_dims(
        image, axis=0
    )  # * expand the image to be 4D (batch, height, width, channels
    image = tf.cast(image, tf.float32)  # * convert the image to float32
    return image


def enhance_image(model: tf.keras.Model, image: tf.Tensor) -> tf.Tensor:
    image = model(prepare_for_model(image))
    return tf.squeeze(image)
