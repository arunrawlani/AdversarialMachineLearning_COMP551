from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
import numpy as np
import os
from six.moves import xrange

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

#from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval

from cleverhans.attacks import jsma
from cleverhans.attacks_tf import jacobian_graph
from cleverhans.utils import other_classes #, pair_visual #, grid_visual #, cnn_model

from keras.datasets import cifar10
from keras import backend
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

import matplotlib.pyplot as plt

from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
from keras.applications.vgg19 import VGG19
from keras.models import Model



from distutils.version import LooseVersion
if LooseVersion(keras.__version__) >= LooseVersion('2.0.0'):
    from keras.layers import Conv2D
else:
    from keras.layers import Convolution2D

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
flags.DEFINE_boolean('viz_enabled', True, 'Enable sample visualization.')
flags.DEFINE_integer('nb_epochs', 15, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_integer('nb_classes', 10, 'Number of classification classes')
flags.DEFINE_integer('img_rows', 96, 'Input row dimension')
flags.DEFINE_integer('img_cols', 96, 'Input column dimension')
flags.DEFINE_integer('nb_channels', 3, 'Nb of color channels in the input.')
flags.DEFINE_integer('nb_filters', 64, 'Number of convolutional filter to use')
flags.DEFINE_integer('nb_pool', 2, 'Size of pooling area for max pooling')
flags.DEFINE_integer('source_samples', 10, 'Nb of test set examples to attack')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for training')


##############################
#Two new functions
#############################
def vgg19(input_shape):
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # add a global spatial average pooling layer
    x = base_model.output
    x = MaxPooling2D()(x)
    # let's add a fully-connected layer
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    #x = Dense(512, activation='relu')(x)
    #x = Dropout(0.5)(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(10, activation='softmax')(x)
    
    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)
    
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # compile the model (should be done *after* setting layers to non-trainable)
    # model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
#return predictions


def data_stl10():
    """
        Preprocess STL dataset
        :return:
        """
    
    # These values are specific to CIFAR10
    img_rows = 96
    img_cols = 96
    nb_classes = 10
    
    # the data, shuffled and split between train and test sets
    #(X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = np.load('x_stl10_train.npy')
    y_train = np.load('y_stl10_train.npy') - 1
    X_test = np.load('x_stl10_test.npy')
    y_test = np.load('y_stl10_test.npy') - 1
    if keras.backend.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    # np.save("cifar10_legitimate.npy",X_test)
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test


#getting the grid visualization
def grid_visual(data):
    """
        This function displays a grid of images to show full misclassification
        :param data: grid data of the form;
        [nb_classes : nb_classes : img_rows : img_cols : nb_channels]
        :return: if necessary, the matplot figure to reuse
        """
    
    # Ensure interactive mode is disabled and initialize our graph
    plt.ioff()
    figure = plt.figure()
    figure.canvas.set_window_title('Cleverhans: Grid Visualization')
    
    # Add the images to the plot
    num_cols = data.shape[0]
    num_rows = data.shape[1]
    num_channels = data.shape[4]
    current_row = 0
    for y in xrange(num_rows):
        for x in xrange(num_cols):
            figure.add_subplot(num_cols, num_rows, (x+1)+(y*num_rows))
            plt.axis('off')
            
            if num_channels == 1:
                plt.imshow(data[x, y, :, :, 0], cmap='gray')
            else:
                plt.imshow(data[x, y, :, :, :])

    # Draw the plot and return
    plt.savefig("grid_cifar")
    return figure

#getting CIFAR10 dataset and preprocessing it
def data_cifar10():
    """
        Preprocess CIFAR10 dataset
        :return:
        """
    
    # These values are specific to CIFAR10
    img_rows = 32
    img_cols = 32
    nb_classes = 10
    
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    if keras.backend.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test


#conv_2d
def conv_2d(filters, kernel_shape, strides, padding):
    """
        Defines the right convolutional layer according to the
        version of Keras that is installed.
        :param filters: (required integer) the dimensionality of the output
        space (i.e. the number output of filters in the
        convolution)
        :param kernel_shape: (required tuple or list of 2 integers) specifies
        the strides of the convolution along the width and
        height.
        :param padding: (required string) can be either 'valid' (no padding around
        input or feature map) or 'same' (pad to ensure that the
        output feature map size is identical to the layer input)
        :return: the Keras layer
        """
    if LooseVersion(keras.__version__) >= LooseVersion('2.0.0'):
        return Conv2D(filters=filters, kernel_size=kernel_shape,
                      strides=strides, padding=padding)
    else:
        return Convolution2D(filters, kernel_shape[0], kernel_shape[1],
                             subsample=strides, border_mode=padding)


# the cnn_model used
def cnn_model(logits=False, input_ph=None, img_rows=28, img_cols=28,
              channels=1, nb_filters=64, nb_classes=10):
    """
        Defines a CNN model using Keras sequential model
        :param logits: If set to False, returns a Keras model, otherwise will also
        return logits tensor
        :param input_ph: The TensorFlow tensor for the input
        (needed if returning logits)
        ("ph" stands for placeholder but it need not actually be a
        placeholder)
        :param img_rows: number of row in the image
        :param img_cols: number of columns in the image
        :param channels: number of color channels (e.g., 1 for MNIST)
        :param nb_filters: number of convolutional filters per layer
        :param nb_classes: the number of output classes
        :return:
        """
    model = Sequential()
    
    # Define the layers successively (convolution layers are version dependent)
    if keras.backend.image_dim_ordering() == 'th':
        input_shape = (channels, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, channels)
    
    layers = [Dropout(0.2, input_shape=input_shape),
              conv_2d(nb_filters, (8, 8), (2, 2), "same"),
              Activation('relu'),
              conv_2d((nb_filters * 2), (6, 6), (2, 2), "valid"),
              Activation('relu'),
              conv_2d((nb_filters * 2), (5, 5), (1, 1), "valid"),
              Activation('relu'),
              Dropout(0.5),
              Flatten(),
              Dense(nb_classes)]

    for layer in layers:
        model.add(layer)
    
    if logits:
        logits_tensor = model(input_ph)
    model.add(Activation('softmax'))
    
    if logits:
        return model, logits_tensor
    else:
        return model

def main(argv=None):
    """
    MNIST tutorial for the Jacobian-based saliency map approach (JSMA)
    :return:
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    ###########################################################################
    # Define the dataset and model
    ###########################################################################

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' "
              "to 'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)
    print("Created TensorFlow session and set Keras backend.")

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_stl10()
    print("Loaded STL10 test data")
    #print("Loaded CIFAR10 test data")
    #print("Loaded MNIST test data.")

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 96, 96, 3))
    #x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Define TF model graph (MNIST)
    #model = cnn_model()
    #predictions = model(x)
    #print("Defined TensorFlow model graph.")

    # Define TF model graph
    model = vgg19((96,96,3))
    predictions = model(x)
    print("Defined TensorFlow model graph.")

    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################

    # Train an MNIST model if it does not exist in the train_dir folder
    saver = tf.train.Saver()
    save_path = os.path.join(FLAGS.train_dir, FLAGS.filename)
    if os.path.isfile(save_path):
        saver.restore(sess, os.path.join(FLAGS.train_dir, FLAGS.filename))
    else:
        train_params = {
            'nb_epochs': FLAGS.nb_epochs,
            'batch_size': FLAGS.batch_size,
            'learning_rate': FLAGS.learning_rate
        }
        model_train(sess, x, y, predictions, X_train, Y_train,
                    args=train_params)
        saver.save(sess, save_path)

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': FLAGS.batch_size}
    accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                          args=eval_params)
    # assert X_test.shape[0] == 10000, X_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))

    ###########################################################################
    # Craft adversarial examples using the Jacobian-based saliency map approach
    ###########################################################################
    print('Crafting ' + str(FLAGS.source_samples) + ' * ' +
          str(FLAGS.nb_classes-1) + ' adversarial examples')

    # This array indicates whether an adversarial example was found for each
    # test set sample and target class
    results = np.zeros((FLAGS.nb_classes, FLAGS.source_samples), dtype='i')

    # This array contains the fraction of perturbed features for each test set
    # sample and target class
    perturbations = np.zeros((FLAGS.nb_classes, FLAGS.source_samples),
                             dtype='f')

    # Define the TF graph for the model's Jacobian
    grads = jacobian_graph(predictions, x, FLAGS.nb_classes)

    # Initialize our array for grid visualization
    grid_shape = (FLAGS.nb_classes,
                  FLAGS.nb_classes,
                  FLAGS.img_rows,
                  FLAGS.img_cols,
                  FLAGS.nb_channels)
    grid_viz_data = np.zeros(grid_shape, dtype='f')

    # Loop over the samples we want to perturb into adversarial examples
    for sample_ind in xrange(0, FLAGS.source_samples):
        # We want to find an adversarial example for each possible target class
        # (i.e. all classes that differ from the label given in the dataset)
        current_class = int(np.argmax(Y_test[sample_ind]))
        target_classes = other_classes(FLAGS.nb_classes, current_class)

        # For the grid visualization, keep original images along the diagonal
        grid_viz_data[current_class, current_class, :, :, :] = np.reshape(
                X_test[sample_ind:(sample_ind+1)],
                (FLAGS.img_rows, FLAGS.img_cols, FLAGS.nb_channels))

        # Loop over all target classes
        for target in target_classes:
            print('--------------------------------------')
            print('Creating adv. example for target class ' + str(target))

            # This call runs the Jacobian-based saliency map approach
            adv_x, res, percent_perturb = jsma(sess, x, predictions, grads,
                                               X_test[sample_ind:
                                                      (sample_ind+1)],
                                               target, theta=1, gamma=0.1,
                                               increase=True, back='tf',
                                               clip_min=0, clip_max=1)

            # Display the original and adversarial images side-by-side
#            if FLAGS.viz_enabled:
#                if 'figure' not in vars():
#                        figure = pair_visual(
#                                np.reshape(X_test[sample_ind:(sample_ind+1)],
#                                           (FLAGS.img_rows, FLAGS.img_cols, FLAGS.nb_channels)),
#                                np.reshape(adv_x,
#                                           (FLAGS.img_rows, FLAGS.img_cols, FLAGS.nb_channels)))
#                else:
#                    figure = pair_visual(
#                            np.reshape(X_test[sample_ind:(sample_ind+1)],
#                                       (FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels)),
#                            np.reshape(adv_x, (FLAGS.img_rows,
#                                       FLAGS.img_cols,FLAGS.nb_channels)), figure)

            # Add our adversarial example to our grid data
            grid_viz_data[target, current_class, :, :, :] = np.reshape(
                    adv_x, (FLAGS.img_rows, FLAGS.img_cols, FLAGS.nb_channels))
            sum = np.sum(adv_x)
            if sum == 0.0 or sum == 0:
                print('HEY')
                quit()
            # Update the arrays for later analysis
            results[target, sample_ind] = res
            perturbations[target, sample_ind] = percent_perturb

    # Compute the number of adversarial examples that were successfuly found
    nb_targets_tried = ((FLAGS.nb_classes - 1) * FLAGS.source_samples)
    succ_rate = float(np.sum(results)) / nb_targets_tried
    print('Avg. rate of successful adv. examples {0:.2f}'.format(succ_rate))

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(perturbations)
    print('Avg. rate of perturbed features {0:.2f}'.format(percent_perturbed))

    # Compute the average distortion introduced for successful samples only
    percent_perturb_succ = np.mean(perturbations * (results == 1))
    print('Avg. rate of perturbed features for successful '
          'adversarial examples {0:.2f}'.format(percent_perturb_succ))

    # Close TF session
    sess.close()

    # Finally, block & display a grid of all the adversarial examples
    if FLAGS.viz_enabled:
        _ = grid_visual(grid_viz_data)

if __name__ == '__main__':
    app.run()
