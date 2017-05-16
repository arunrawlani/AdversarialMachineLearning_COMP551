from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import keras
from keras import backend
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
from keras.applications.vgg19 import VGG19

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from utils_mnist import data_mnist

from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'cifar10.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 15, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 64, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for training')

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
    np.save("cifar10_legitimate.npy",X_test)
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test
    
def data_stl10():
    """
    Preprocess CIFAR10 dataset
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


def main(argv=None):
    """
    CIFAR10 cleverhans tutorial
    :return:
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Get CIFAR10 test data
    X_train, Y_train, X_test, Y_test = data_cifar10()

    assert Y_train.shape[1] == 10.
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Define TF model graph
    model = cnn_model(img_rows=32, img_cols=32, channels=3)
    # model = vgg19(input_shape=(96,96,3))
    predictions = model(x)
    print(type(predictions))
    print("Defined TensorFlow model graph.")

    def evaluate():
        # Evaluate the accuracy of the CIFAR10 model on legitimate test examples
        eval_params = {'batch_size': FLAGS.batch_size}
        accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                              args=eval_params)
        # assert X_test.shape[0] == 10000, X_test.shape
        # train_accuracy = model_eval(sess, x, y, predictions, X_train, Y_train,
        #                      args=eval_params)
        # print('Train accuracy on legitimate train examples: ' + str(train_accuracy))
        print('Test accuracy on legitimate test examples: ' + str(accuracy))

    # Train an CIFAR10 model
    train_params = {
        'nb_epochs': FLAGS.nb_epochs,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate
    }
    model_train(sess, x, y, predictions, X_train, Y_train,
                evaluate=evaluate, args=train_params)

    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    adv_x = fgsm(x, predictions, eps=0.05)
    print(type(adv_x))
    print(adv_x.get_shape())
    eval_params = {'batch_size': FLAGS.batch_size}
    X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test], args=eval_params)
    print(type(X_test_adv))
    print(X_test_adv.shape)
    np.save("adversarial_samples.npy",X_test_adv)
    np.save("adversarial_true_labels.npy", Y_test)
    #assert X_test_adv.shape[0] == 10000, X_test_adv.shape

    # Evaluate the accuracy of the CIFAR10 model on adversarial examples
    accuracy = model_eval(sess, x, y, predictions, X_test_adv, Y_test,
                          args=eval_params)
    print('Test accuracy on adversarial examples: ' + str(accuracy))

    print("Repeating the process, using adversarial training")
    # Redefine TF model graph
    model_2 = cnn_model(img_rows=32, img_cols=32, channels=3)
    # model_2 = vgg19(input_shape=(96,96,3))
    predictions_2 = model_2(x)
    adv_x_2 = fgsm(x, predictions_2, eps=0.05)
    print(type(adv_x_2))
    predictions_2_adv = model_2(adv_x_2)

    def evaluate_2():
        # Evaluate the accuracy of the adversarialy trained CIFAR10 model on
        # legitimate test examples
        eval_params = {'batch_size': FLAGS.batch_size}
        accuracy = model_eval(sess, x, y, predictions_2, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on legitimate test examples: ' + str(accuracy))

        # Evaluate the accuracy of the adversarially trained CIFAR10 model on
        # adversarial examples
        accuracy_adv = model_eval(sess, x, y, predictions_2_adv, X_test,
                                  Y_test, args=eval_params)
        print('Test accuracy on adversarial examples: ' + str(accuracy_adv))

    # Perform adversarial training
    model_train(sess, x, y, predictions_2, X_train, Y_train,
                predictions_adv=predictions_2_adv, evaluate=evaluate_2,
                args=train_params)

    # Evaluate the accuracy of the CIFAR10 model on adversarial examples
    accuracy = model_eval(sess, x, y, predictions_2_adv, X_test, Y_test,
                          args=eval_params)
    print('Test accuracy on adversarial examples: ' + str(accuracy))



if __name__ == '__main__':
    app.run()
