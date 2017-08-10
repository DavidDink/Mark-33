"""
Builds the Neural Network for the Building Management System

Summary of available functions:

# evaluations, use inputs
# Compute input images and labels for training. If you would like to run

Built Based on this blog:
https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb
"""

import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import confusion_matrix
from datetime import timedelta
import math


# ===============================
# Configuration of Neural Network
# ===============================

# First Convolutional Layer
first_filter_size = 5
num_first_filters = 16

# Second Convolutional Layer
second_filter_size = 5
num_second_filter = 36

# Last layer is a fully-connected layer
fc_size = 128 # Number of nuerons in fully-connected layer



# ===============================
# Load Data
# ===============================

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot =True)

data.test.cls = np.argmax(data.test.labels, axis=1)


# ===============================
# Data Dimensions
# ===============================

# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of input channels for the images: 1 channel for gray-scale
num_channels = 1

# Number of output classes, in this case one class for each of 10 digits
num_classes = 10

"""
Tensorflow graph consists of the following parts which will be shown below:

1. Placeholder variables used for inputting data to the graph
2. Variables that are going to be optimized so as to make the convolutional network perform better
3. The mathematical formulas for the convolutional network.
4. A cost measure that can be used to guide the optimization of the variables.
5. An optimization method which updates the variables. 

"""



# ===============================
# Defining Tensorflow Variables.
# ===============================
"""
NOTE: We are not initializing the network. 
We are defining variables that will be used later. 
"""

def new_weights(shape): #Creates random weights
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length): #Creates random biases
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input,               # The previous layer
                   num_input_channels,  # Num. channels in prev. layer
                   filter_size,         # Width and height of each filter
                   num_filters,         # Number of filters
                   use_pooling=True):   # Boolean for whether we want to use 2x2 max-pooling


    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights, # Gives the weight
                         strides=[1, 1, 1, 1], # Means it moves
                         padding='SAME') # Size of Input = Size of Output

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.

def new_fc_layer(input,          # Output of the previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer # So we can continue building the graph


# ===============================
# Placeholder variables.
# ===============================
"""
Placeholder variables are inputs to the TensorFlow graph
that we may change each time we execute the graph. 
"""

# Data type is set to float 32 and the shape is set to [None, img-size_flat]
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

# Since the convolutional layers expect X to be encoded as a 4-dim
# tensor so we have to reshape it so its shape is instead [num_images,
# img_width, img_height, num_channels]. Note that img_width == img_height
# == img_size and num_images can be inferred automatically by using -1
# for the size of the first dimension. So the reshape operation is:

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

# Placeholder variable for the output labels.
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

# We could also have a placeholder variable for the class-number, but we will
# instead calculate it using argmax. Note that this is a TensorFlow operator
# so nothing is calculated at this point.

y_true_cls = tf.argmax(y_true, dimension=1)


# ===============================
# Creating the Convolutional Layers
# ===============================
"""Creating the first convolutional layer. 
It takes x_image as inputs and creates number of filters1. 
"""
layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

"""Creating the second convolutional layer.
It takes output of the first layer as inputs and creates number of filters2.   
"""

layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

"""Flatten Layer
Last layers of a convolutional neural network is a fully-connected layer.
To use the result of the first two layers to the last layer, we need a 
layer that will reshape or flatten the output from the second layer.
"""

layer_flat, num_features = flatten_layer(layer_conv2)


"""Second Last layer - Fully-Connected Layer 1
"""

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True) #relu is used so we can learn non-linear relations

"""Last layer - Fully-Connected Layer2
Outputs vector of length 10 for determining which class the input image belongs
to.
"""

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)




# ===============================
# Cleaning up the Output
# ===============================
"""
Congragulations. You finally got the output from the neural network.
One small step for us, one great step for humanity. However, we're 
not done yet. Usually, the output of the neural network is a bit rough
and difficult to interpret because the numbers may be very small or large.
Thus, we will normalize the output, limiting it between zero and one
and the 10 elements sum to one.

Softmax function helps us with this.
"""

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)


# ===============================
# Updating the weights of the Neural Network
# ===============================
"""
To update the neural network properly, weneed to know how well the 
model currently performs by comparing the predicted output of the model
y_pred to the desired output y_true.

Cross-entropy is a continuous function that is always positive and 
if the predicted output of the model exactly matches the desired output
then the cross-entropy equals zero. The goal of optimization is 
therefore to minimize the cross-entropy so it gets as close to zero 
as possible by changing the variables of the network layers. Cross-
entropy is commonly used in classification problems. 

TensorFlow has a built-in function for calculating the cross-entropy. 
Note that the function calculates the softmax internally so we must 
use the output of layer_fc2 directly rather than y_pred which has 
already had the softmax applied.
"""

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)

cost = tf.reduce_mean(cross_entropy) #Takes the average of the cross-
                                     #entropy for all the image classifications.
                                     #This is used to create a single singular
                                     #scalar value for guidance.


# ===============================
# Optimization Method
# ===============================
"""
Used to minimize the cost measure
"""

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
#AdamOptimizer is an advanced form of Gradient Descent.



# ===============================
# Performance Measure
# ===============================
"""
Purpose: to know the progress of increasing accuracy of the neural network.
"""

# Vector of booleans whether the predicted class equals the true class
# of each image.

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf,cast(correct_prediction, tf.float32))



# ===============================
# Running the tensorflow - It's pizza time.
# ===============================

session = tf.Session()
session.run(tf.global_variables_initializer()) #Initializing weights and biases

# ===============================
# Optimization Iterations
# ===============================

train_batch_size = 64 # This is the number of samples for episode of
                  # training.

                  # If you have a better computer spec than my
                  # 2013 Macbook Air, you may increase the size
                  # of 'train_batch_size'


""""
Function for performing a number of optimization iteration so as to gradually
improve the network layers.

In each iteration, a new batch of data is selected from the 
training-set and then TensorFlow executes the optimizer using those 
training samples. The progress is printed every 100 iterations.
"""


# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# Helper-function to plot confusion matrix
def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.test.cls

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# =================================
# Testing the Neural Network
# =================================

# Split the test-set into smaller batchesof this size.
test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

print_test_accuracy()



# ========================
# Visualization of the Neural Network
# ========================

"""Visulation of Weights and Layers
"""


def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

"""Visualization of the output
"""

def plot_conv_layer(layer, image):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the output of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.

    # Create a feed-dict containing just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    feed_dict = {x: [image]}

    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    values = session.run(layer, feed_dict=feed_dict)

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i < num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

""" Visulization of input image
"""

def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')

    plt.show()

