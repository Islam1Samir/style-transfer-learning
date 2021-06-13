import tensorflow as tf
import utilities
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K

def tensor_to_image(tensor):
  '''converts a tensor to an image'''
  tensor_shape = tf.shape(tensor)
  number_elem_shape = tf.shape(tensor_shape)
  print(number_elem_shape)
  if number_elem_shape > 3:
    assert tensor_shape[0] == 1
    tensor = tensor[0]
    print('yas')
  return tf.keras.preprocessing.image.array_to_img(tensor)


def preprocess_image(image):
  '''preprocesses a given image to use with Inception model'''
  image = tf.cast(image, dtype=tf.float32)
  image = (image / 127.5) - 1.0

  return image


def inception_model(layer_names):

    inception = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')
    inception.trainable = False
    output_layers = [inception.get_layer(name).output for name in layer_names]
    model = tf.keras.models.Model(inputs=inception.input, outputs=output_layers)
    return model

def get_style_loss(outputs, targets):
    style_loss = tf.reduce_mean(tf.square(outputs - targets))
    return style_loss

def get_content_loss(outputs, targets):
    content_loss = 0.5 * tf.reduce_sum(tf.square(targets - targets))
    return content_loss


def gram_matrix(input_tensor):

    gram = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)

    input_shape = tf.shape(input_tensor)
    height = input_shape[1]
    width = input_shape[2]

    num_locations = tf.cast(height * width, tf.float32)
    scaled_gram = gram / num_locations

    return scaled_gram

def get_style_image_features(image):
    preprocessed_style_image = preprocess_image(image)

    outputs = inception(preprocessed_style_image)
    style_outputs = outputs[1:]

    gram_style_features = [gram_matrix(style_layer) for style_layer in style_outputs]
    return gram_style_features


def get_content_image_features(image):

    preprocessed_content_image = preprocess_image(image)

    outputs = inception(preprocessed_content_image)

    content_outputs = outputs[:1]
    return content_outputs


def get_style_content_loss(style_targets, style_outputs, content_targets,
                           content_outputs, style_weight, content_weight):

    style_loss = tf.add_n([get_style_loss(style_output, style_target)
                           for style_output, style_target in zip(style_outputs, style_targets)])

    content_loss = tf.add_n([get_content_loss(content_output, content_target)
                             for content_output, content_target in zip(content_outputs, content_targets)])


    style_loss = style_loss * style_weight / 5

    content_loss = content_loss * content_weight / 1

    total_loss = style_loss + content_loss
    return total_loss


def calculate_gradients(image, style_targets, content_targets,
                        style_weight, content_weight):

    with tf.GradientTape() as tape:
        style_features = get_style_image_features(image)

        content_features = get_content_image_features(image)

        loss = get_style_content_loss(style_targets, style_features, content_targets,
                                      content_features, style_weight, content_weight)
        loss += var_weight*tf.image.total_variation(image)

    gradients = tape.gradient(loss, image)

    return gradients


def update_image_with_style(image, style_targets, content_targets, style_weight,
                            content_weight, optimizer):

  gradients = calculate_gradients(image, style_targets, content_targets,
                                  style_weight, content_weight)

  optimizer.apply_gradients([(gradients, image)])


  image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0))

def fit_style_transfer(style_image, content_image, style_weight=1e-2, content_weight=1e-4,
                         optimizer='adam', epochs=1, steps_per_epoch=1):

      images = []
      step = 0

      style_targets = get_style_image_features(style_image)

      content_targets = get_content_image_features(content_image)

      generated_image = tf.cast(content_image, dtype=tf.float32)
      generated_image = tf.Variable(generated_image)

      images.append(content_image)

      for n in range(epochs):
          for m in range(steps_per_epoch):
              step += 1
              update_image_with_style(generated_image, style_targets, content_targets,
                                      style_weight, content_weight, optimizer)


              print(".", end='')
              if (m + 1) % 50 == 0:
                  images.append(generated_image)

          images.append(generated_image)
          print("Train step: {}".format(step))

      generated_image = tf.cast(generated_image, dtype=tf.uint8)

      return generated_image, images


content_image =utilities.load_image('pyramids.jpg')
style_image = utilities.load_image('style image.jpeg')

output_layers = ['conv2d_93', 'conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4']

inception = inception_model(output_layers)

style_weight =  1
content_weight = 1e-25
var_weight = 0

adam = tf.optimizers.Adam(
    tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=80.0, decay_steps=100, decay_rate=0.80
    )
)

stylized_image, display_images = fit_style_transfer(style_image=style_image, content_image=content_image,
                                                    style_weight=style_weight, content_weight=content_weight,
                                                    optimizer=adam, epochs=5, steps_per_epoch=100)

display_images = [tf.squeeze(im, axis=0).numpy() for im in display_images]
utilities.create_gif(display_images)