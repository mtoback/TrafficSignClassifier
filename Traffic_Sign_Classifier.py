
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission, if necessary. Sections that begin with **'Implementation'** in the header indicate where you should begin your implementation for your project. Note that some sections of implementation are optional, and will be marked with **'Optional'** in the header.
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[ ]:
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import scipy.ndimage
from skimage import exposure

# Load pickled data
training_file = "train.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below.

# In[ ]:

### Replace each question mark with the appropriate value.
# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = train['labels'].max() + 1
from sklearn.utils import shuffle
# shuffle the data before you take the validation data out
X_train, y_train = shuffle(X_train, y_train)
# Clean up images
threshold = 30
maxIntensity = 255.0 # depends on dtype of image data
# Parameters for manipulating image data
''' take this out for now
for index in np.arange(len(X_train)):
    avg_color = np.average([np.average(np.average(X_train[index], axis=0)),
     np.average(np.average(X_train[index], axis=1)),
     np.average(np.average(X_train[index], axis=2))])
    if avg_color < threshold:
        X_train[index] = exposure.rescale_intensity(X_train[index])
'''
# increase the amount of training data
hist, n_bins = np.histogram(y_train, bins=np.max(y_train))
center = (n_bins[:-1] + n_bins[1:]) / 2
width = 0.7 * (n_bins[1] - n_bins[0])
print("before leveling=",hist)
#plt.bar(center, hist, align='center', width=width)
#plt.show()
weak_classes = [0, 6, 14, 15, 16, 19, 20, 21, 22, 23, 24, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 36, 37, 39, 40, 41]
need_more = 0
for weak_class in weak_classes:
    need_more += 2*hist[weak_class]
shape = np.shape(X_train)
temp_train = np.zeros([need_more + shape[0],shape[1],shape[2],shape[3]])
temp_ytrain = np.zeros([need_more + shape[0]])
next = shape[0] - 1
for index in np.arange(n_train):
    temp_train[index] = X_train[index]
    temp_ytrain[index] = y_train[index]
    if y_train[index] in weak_classes:
        temp_img = scipy.ndimage.interpolation.shift(X_train[index], 
                    [random.randrange(-2, 2), random.randrange(-2, 2), 0])
        temp_img = scipy.ndimage.interpolation.rotate(temp_img,
                    random.randrange(-10, 10), reshape=False)
        temp_train[next + 1] = temp_img
        temp_ytrain[next +1] = y_train[index]
        temp_img = scipy.ndimage.interpolation.shift(X_train[index], 
                    [random.randrange(-2, 2), random.randrange(-2, 2), 0])
        temp_img = scipy.ndimage.interpolation.rotate(temp_img,
                    random.randrange(-10, 10), reshape=False)
        temp_train[next+2] = temp_img
        temp_ytrain[next+2] = y_train[index]
        next = next + 2
X_train = temp_train
y_train = temp_ytrain
hist, n_bins = np.histogram(y_train, bins=np.max(y_train))
print("after leveling=",hist)
center = (n_bins[:-1] + n_bins[1:]) / 2
width = 0.7 * (n_bins[1] - n_bins[0])
plt.bar(center, hist, align='center', width=width)
plt.show()
# now augment the data and normalize it
shape = np.shape(X_train)
temp_train = np.zeros([3*shape[0],shape[1],shape[2],shape[3]])
temp_ytrain = np.zeros([3*shape[0]])
for index in np.arange(n_train):
    temp_train[3*index] = X_train[index]
    temp_ytrain[3*index] = y_train[index]
    temp_img = scipy.ndimage.interpolation.shift(X_train[index], 
                [random.randrange(-2, 2), random.randrange(-2, 2), 0])
    temp_img = scipy.ndimage.interpolation.rotate(temp_img,
                random.randrange(-10, 10), reshape=False)
    temp_train[3*index + 1] = temp_img
    temp_ytrain[3*index +1] = y_train[index]
    temp_img = scipy.ndimage.interpolation.shift(X_train[index], 
                [random.randrange(-2, 2), random.randrange(-2, 2), 0])
    temp_img = scipy.ndimage.interpolation.rotate(temp_img,
                random.randrange(-10, 10), reshape=False)
    temp_train[3*index+2] = temp_img
    temp_ytrain[3*index+2] = y_train[index]
# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc.
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections.

# In[ ]:

### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
# Visualizations will be shown in the notebook.
#get_ipython().magic('matplotlib inline')

index = random.randint(0, len(X_train))
image = X_train[index]
#plt.imshow(image)
#plt.show()
# normalize the training data... some of it is really dark/light 
# so better to normalize based on individual ranges
X_train = temp_train
y_train = temp_ytrain
for index in range(len(X_train)):
    max = float(np.amax(X_train[index]))
    min = float(np.amin(X_train[index]))
    X_train[index] = -1+ 2*(X_train[index] - min)/(max-min)
    max = float(np.amax(X_train[index]))
    min = float(np.amin(X_train[index]))
    pass
for index in range(len(X_test)):
    max = float(np.amax(X_test[index]))
    min = float(np.amin(X_test[index]))
    X_test[index] = -1+ 2*(X_train[index] - min)/(max-min)
# extract the validation data
X_train, y_train = shuffle(X_train, y_train)
n_validation = int((n_train*2)/10)
X_validation = X_train[0:n_validation,:,:,:]
y_validation = y_train[0:n_validation]
X_train = X_train[n_validation:,:,:,:]
y_train = y_train[n_validation:]
# recalculate number of training examples
n_train = len(X_train)

# clean up colors on test data and normalize it
'''
for index in np.arange(len(X_test)):
    avg_color = np.average([np.average(np.average(X_test[index], axis=0)),
    np.average(np.average(X_test[index], axis=1)),
    np.average(np.average(X_test[index], axis=2))])
    if avg_color < threshold:
        X_test[index] = exposure.rescale_intensity(X_test[index])
'''
print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)



# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.
# 
# **NOTE:** The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play!

# ### Implementation
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project. Once you have completed your implementation and are satisfied with the results, be sure to thoroughly answer the questions that follow.

# In[ ]:

### Preprocess the data here.
### Feel free to use as many code cells as needed.


# ## Setup TensorFlow
# The `EPOCH` and `BATCH_SIZE` values affect the training speed and model accuracy.
# 
# You do not need to modify this section.

# In[ ]:

import tensorflow as tf

EPOCHS = 20 # it drops like a rock after 11
BATCH_SIZE = 128
dropout = 0.74  # Dropout, probability to keep units


# ## SOLUTION: Implement LeNet-5
# Implement the [LeNet-5](http://yann.lecun.com/exdb/lenet/) neural network architecture.
# 
# This is the only cell you need to edit.
# ### Input
# The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Since MNIST images are grayscale, C is 1 in this case.
# 
# ### Architecture
# **Layer 1: Convolutional.** The output shape should be 28x28x6.
# 
# **Activation.** Your choice of activation function.
# 
# **Pooling.** The output shape should be 14x14x6.
# 
# **Layer 2: Convolutional.** The output shape should be 10x10x16.
# 
# **Activation.** Your choice of activation function.
# 
# **Pooling.** The output shape should be 5x5x16.
# 
# **Flatten.** Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using `tf.contrib.layers.flatten`, which is already imported for you.
# 
# **Layer 3: Fully Connected.** This should have 120 outputs.
# 
# **Activation.** Your choice of activation function.
# 
# **Layer 4: Fully Connected.** This should have 84 outputs.
# 
# **Activation.** Your choice of activation function.
# 
# **Layer 5: Fully Connected (Logits).** This should have 10 outputs.
# 
# ### Output
# Return the result of the 2nd fully connected layer.

# In[ ]:

from tensorflow.contrib.layers import flatten
from tensorflow.core.protobuf.config_pb2 import ConfigProto
mu = 0
sigma = 0.1
conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma), name="conv1_W")
conv1_b = tf.Variable(tf.zeros(6), name="conv1_b")
conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma), name="conv2_W")
conv2_b = tf.Variable(tf.zeros(16), name="conv2_b")
fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma), name="fc1_W")
fc1_b = tf.Variable(tf.zeros(120), name="fc1_b")
fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma), name="fc2_W")
fc2_b  = tf.Variable(tf.zeros(84), name="fc2_b")
fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma), name="fc3_W")
fc3_b  = tf.Variable(tf.zeros(n_classes), name="fc3_b")

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation of Layer 1
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # Activation.
    fc1    = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Activation.
    fc2    = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)

    # Layer 5: Fully Connected. Input = 84. Output = n_classes.
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits


# ## Features and Labels
# Train LeNet to classify [MNIST](http://yann.lecun.com/exdb/mnist/) data.
# 
# `x` is a placeholder for a batch of input images.
# `y` is a placeholder for a batch of output labels.
# 
# In[ ]:

x = tf.placeholder(tf.float32, (None, 32, 32, 3), name="x")
y = tf.placeholder(tf.int32, (None), name="y")
num_labels = n_classes
sparse_labels = tf.reshape(y, [-1, 1])
derived_size = tf.shape(sparse_labels)[0]
indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
concated = tf.concat(1, [indices, sparse_labels])
outshape = tf.concat(0, [tf.reshape(derived_size, [1]), tf.reshape(num_labels, [1])])
one_hot_y = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)

# ## Training Pipeline
# Create a training pipeline that uses the model to classify MNIST data.
# 
# You do not need to modify this section.

# In[ ]:

rate = 0.0005

logits = LeNet(x)
beta = 0.0012
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy +
    beta*tf.nn.l2_loss(conv1_W) +
    beta*tf.nn.l2_loss(conv2_W) +
    beta*tf.nn.l2_loss(fc1_W) +
    beta*tf.nn.l2_loss(fc2_W) +
    beta*tf.nn.l2_loss(fc2_W))
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


# ## Model Evaluation
# Evaluate how well the loss and accuracy of the model for a given dataset.
# 
# You do not need to modify this section.

# In[ ]:

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# ## Train the Model
# Run the training data through the training pipeline to train the model.
# 
# Before each epoch, shuffle the training set.
# 
# After each epoch, measure the loss and accuracy of the validation set.
# 
# Save the model after training.
# 
# You do not need to modify this section.

# In[ ]:
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    last_validation_accuracy = 0
    writer = tf.train.SummaryWriter("logs", graph=tf.get_default_graph())
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        # if we're not improving, we're done
        if(last_validation_accuracy > validation_accuracy):
            break
        else:
            last_validation_accuracy = validation_accuracy
        
    saver.save(sess, './lenet')
    print("Model saved")


# ## Evaluate the Model
# Once you are completely satisfied with your model, evaluate the performance of the model on the test set.
# 
# Be sure to only do this once!
# 
# If you were to measure the performance of your trained model on the test set, then improve your model, and then measure the performance of your model on the test set again, that would invalidate your test results. You wouldn't get a true measure of how well your model would perform against real data.
# 
# You do not need to modify this section.

# In[ ]:
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


# ### Question 1 
# 
# _Describe how you preprocessed the data. Why did you choose that technique?_

# **Answer:**

# In[ ]:

### Generate additional data (OPTIONAL!)
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.


# ### Question 2
# 
# _Describe how you set up the training, validation and testing data for your model. **Optional**: If you generated additional data, how did you generate the data? Why did you generate the data? What are the differences in the new dataset (with generated data) from the original dataset?_

# **Answer:**

# In[ ]:

### Define your architecture here.
### Feel free to use as many code cells as needed.


# ### Question 3
# 
# _What does your final architecture look like? (Type of model, layers, sizes, connectivity, etc.)  For reference on how to build a deep neural network using TensorFlow, see [Deep Neural Network in TensorFlow
# ](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/b516a270-8600-4f93-a0a3-20dfeabe5da6/concepts/83a3a2a2-a9bd-4b7b-95b0-eb924ab14432) from the classroom._
# 

# **Answer:**

# In[ ]:

### Train your model here.
### Feel free to use as many code cells as needed.


# ### Question 4
# 
# _How did you train your model? (Type of optimizer, batch size, epochs, hyperparameters, etc.)_
# 

# **Answer:**

# ### Question 5
# 
# 
# _What approach did you take in coming up with a solution to this problem? It may have been a process of trial and error, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think this is suitable for the current problem._

# **Answer:**

# ---
# 
# ## Step 3: Test a Model on New Images
# 
# Take several pictures of traffic signs that you find on the web or around you (at least five), and run them through your classifier on your computer to produce example results. The classifier might not recognize some local signs but it could prove interesting nonetheless.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Implementation
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project. Once you have completed your implementation and are satisfied with the results, be sure to thoroughly answer the questions that follow.

# In[ ]:

### Load the images and plot them here.
### Feel free to use as many code cells as needed.


# ### Question 6
# 
# _Choose five candidate images of traffic signs and provide them in the report. Are there any particular qualities of the image(s) that might make classification difficult? It could be helpful to plot the images in the notebook._
# 
# 

# **Answer:**

# In[ ]:

### Run the predictions here.
### Feel free to use as many code cells as needed.


# ### Question 7
# 
# _Is your model able to perform equally well on captured pictures when compared to testing on the dataset? The simplest way to do this check the accuracy of the predictions. For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate._
# 
# _**NOTE:** You could check the accuracy manually by using `signnames.csv` (same directory). This file has a mapping from the class id (0-42) to the corresponding sign name. So, you could take the class id the model outputs, lookup the name in `signnames.csv` and see if it matches the sign from the image._
# 

# **Answer:**

# In[ ]:

### Visualize the softmax probabilities here.
### Feel free to use as many code cells as needed.


# ### Question 8
# 
# *Use the model's softmax probabilities to visualize the **certainty** of its predictions, [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. Which predictions is the model certain of? Uncertain? If the model was incorrect in its initial prediction, does the correct prediction appear in the top k? (k should be 5 at most)*
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# **Answer:**

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# In[ ]:



