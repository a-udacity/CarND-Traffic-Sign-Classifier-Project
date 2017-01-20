
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

# In[100]:

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'traffic-signs-data/train.p'
testing_file = 'traffic-signs-data/test.p'

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
# - `'labels'` is a 2D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below.

# In[101]:

### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[0]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = 43

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc.
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections.

# In[102]:

# load the text labels for the traffic signs
def read_signnames():
    signnames = {}
    with open('signnames.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            signnames[int(row['ClassId'])] = row['SignName']
        
    return signnames


signnames = read_signnames()


# In[103]:

### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
import random
import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 12,
        }

def plot_images_randomly(image_data, labels):
    plt.figure(figsize=(15,15))
    for i in range(15):
        index = random.randint(0, len(image_data))
        plt.subplot(4,4,i+1)
        plt.subplots_adjust(left=0.15)
        plt.imshow(image_data[index])
        plt.text(0, -2, signnames[labels[index]], fontdict=font)
       

plot_images_randomly(X_train, y_train)


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

# In[104]:

### Preprocess the data here.
### Feel free to use as many code cells as needed.
import cv2

def normalize(X):
    X = X / 255.0
    return X


def equalize_channel(X):
    for i in range(0, 3):
        channel = X[:, :, i]
        channel = channel.astype(np.uint8)
        channel = cv2.equalizeHist(channel)
        X[:, :, i] = channel
    return X


def pre_process(X):
    result = []
    for i in range(0, X.shape[0]):
        img = np.array(X[i, :, :, :])
        equalize_channel(img)
        img = normalize(img)
        result.append(img)

    return np.array(result)

X_train = pre_process(X_train)
X_test = pre_process(X_test)


# ### Question 1 
# 
# _Describe how you preprocessed the data. Why did you choose that technique?_

# **Answer:**
# A histogram equalization is performed on the three color channels (RGB) and the pixel values are then (Y) in order to improve contrast and accentuate edges. All color channels are then divided by 255.0 in order to ensure that all samples lie in the range (0.0, 1.0) . This technique was utilized to prevent saturation of neurons in the convolutional neural network

# In[105]:

### Generate data additional data (OPTIONAL!)
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.
from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

assert (len(X_train) == len(y_train))
assert (len(X_validation) == len(y_validation))
assert (len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))


# In[106]:

# Plot images after normalization
plot_images_randomly(X_train, y_train)


# ### Question 2
# 
# _Describe how you set up the training, validation and testing data for your model. **Optional**: If you generated additional data, how did you generate the data? Why did you generate the data? What are the differences in the new dataset (with generated data) from the original dataset?_

# **Answer:**
# The training and validation data was split following the general rule of thumb of 8:2 using the train_test_split function.

# In[107]:

### Define your architecture here.
### Feel free to use as many code cells as needed.
# @See below code cells for implementation


# ### Question 3
# 
# _What does your final architecture look like? (Type of model, layers, sizes, connectivity, etc.)  For reference on how to build a deep neural network using TensorFlow, see [Deep Neural Network in TensorFlow
# ](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/b516a270-8600-4f93-a0a3-20dfeabe5da6/concepts/83a3a2a2-a9bd-4b7b-95b0-eb924ab14432) from the classroom._
# 

# **Answer:**
# LENET-5 Architecture with dropout was used:
# ----------------------------------------
# My final model is a convolutional neural network with two convolutional, two fully connected (dense) layers and 2 dropout. The first convolutional layer has a filter size of 5x5 and a depth of 6 and enables the network to find an optimal color space transformation for the problem.
# The following convolutional layers are alternated with pooling layers with a pooling size of 2x2 and increase in depth for deeper layers from 32 to 128 filters.
# For all layers, rectified linear units (ReLUs) are used as activation functions. The outputs of the pooling layers are combined into an input tensor that is fed into the fully connected layers with 1024 neurons each. 
# In order to improve the robustness of the network, dropout with a keep probability of 0.75 is used on the fully connected layers. Dropout is a regularization technique for reducing overfitting. The technique temporarily drops units (artificial neurons) from the network, along with all of those units' incoming and outgoing connections. The important point was to make sure that the keep_prob was set to 1.0 when evaluating validation accuracy to get the right result.
# The output layer is a fully connected layer with 43 output neurons, which corresponds to the number of different classes in the dataset. Finally, the outputs of the last layers are fed into a softmax function in order to calculate the probabilities for each class.
# The resulting topology is shown below:
# 
# # My Model
# ![My Model](MyModel.png)
# 
# * Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
# * Activation.
# * Layer 2: Convolutional. Output = 10x10x16.
# * Activation.
# * Pooling. Input = 10x10x16. Output = 5x5x16.
# * Flatten. Input = 5x5x16. Output = 400.
# * Layer 3: Fully Connected. Input = 400. Output = 120.
# * Activation.
# * Dropout 1.
# * Layer 4: Fully Connected. Input = 120. Output = 84.
# * Activation.
# * Dropout 2.
# * Layer 5: Fully Connected. Input = 43. Output = 10.
#     
# 

# In[1]:

### Train your model here.
### Feel free to use as many code cells as needed.
def LeNet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Activation.
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)
    
    # Layer 5: Fully Connected. Input = 43. Output = 10.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


# ### Question 4
# 
# _How did you train your model? (Type of optimizer, batch size, epochs, hyperparameters, etc.)_
# 

# **Answer:**
# The Adam Optimizer was used to get the accuracy. ADAM is in reality a SGD algorithm in which the gradient used in each iteration is updated from the previous using a technique based in momenta. Initially I chose a batch size of 128 and Epoch of 10 but found that increasing the batch size to 100 helped me increase my accuracy to over 98%. From the run I found that chosing a Epoch of 65 would have helped me achieve the accuracy. It took about 10 epochs before the validation accuracy converged. A dropout about 50% reduces overfitting. A L2-regularization with dropout further prevents the network to get conditioned badly.
# 
# <pre>
# | Hyper-parameter | Value        |
# |-----------------|--------------|
# |Optimizer Type   |Adam Optimizer|
# |Batch Size       |128           |
# |Number of Epoch  |100           |
# |Learning Rate    |0.001         |
# |Regularization   |Relu          |
# |Drop out         |Keep Prob:0.75|
# |Mean             |0             |
# |Std Deviation    |0.1           |
# </pre>

# In[8]:

import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle

EPOCHS = 100
BATCH_SIZE = 128
save_path = "./trained_model/final"
X_train, y_train = shuffle(X_train, y_train)

keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

drop_out_prob = 0.75
rate = 0.001
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: drop_out_prob })
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer(),feed_dict={keep_prob: drop_out_prob})
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.75})

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, save_path)
    print("Model saved at:" + save_path)


# ### Question 5
# 
# 
# _What approach did you take in coming up with a solution to this problem? It may have been a process of trial and error, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think this is suitable for the current problem._

# **Answer:**
# As per the project description I started my project with LENET architecture from the MNIST data as my starting point. I took the following experimentation below to finally reach an accuracy of over 98%:
# 1. The accuracy decreased when I increased the mean to .2
# 2. The accuracy decreased when I increased the learning rate to 0.01
# 3. The accuracy decreased to 85% when I modified the max pool to average pool.
# 4. Added 2 dropout at keep_prob of 0.5 and EPOCH of 10 but could not get accuracy over 96%
# 5. Added 2 dropout at keep_prob of 0.75 and increasing the number of EPOCH to 100 with Batch size of 128 increased the accuracy to 97.7. I decided to choose this model.
# 
# 
# 
# 

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

# In[109]:

### Load the images and plot them here.
### Feel free to use as many code cells as needed.
get_ipython().magic('matplotlib inline')
import os
from os import listdir
import matplotlib.image as mpimg

image_file_paths = listdir('newImages')

images = []
for path in image_file_paths:
    images.append(mpimg.imread('%s%s' % ('newImages/', path)))
    
new_signs_img = np.zeros([len(images),32,32,3],dtype=np.uint8)
new_signs_label = np.array([15,28,2,31,33],dtype=np.uint8)

for id,img in enumerate(images):
    images[id] = cv2.resize(img,dsize=(32,32))
    new_signs_img[id] = images[id]

pickle_file = 'new_signs.p'
if not os.path.isfile(pickle_file):
    print('Saving data to pickle file...')
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump({
                    'data': new_signs_img,
                    'labels': new_signs_label},
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to:', pickle_file, ':', e)
        raise

print('Data cached in pickle file.')


# In[110]:


with open(pickle_file, mode='rb') as f:
    new_signs = pickle.load(f)
    f.close()

def plot_extra_images(image_data, labels):
    plt.figure(figsize=(15,15))
    for i in range(len(image_data)):
        plt.subplot(1,5,i+1)
        plt.subplots_adjust(left=0.15)
        plt.imshow(image_data[i])
        plt.text(0, -2, signnames[labels[i]], fontdict=font)


X_new = new_signs['data']
y_new = new_signs['labels']

print("                                         Extra Images                 ")
plot_extra_images(X_new, y_new)        


# In[111]:

## Load and normalize the images
get_ipython().magic('matplotlib inline')

X_new = pre_process(X_new)

print("                                     Extra Images after Normalization                ")
plot_extra_images(X_new, y_new) 


# ### Question 6
# 
# _Choose five candidate images of traffic signs and provide them in the report. Are there any particular qualities of the image(s) that might make classification difficult? It could be helpful to plot the images in the notebook._
# 
# 

# **Answer:**
# I took most of the images from the web. The images in between trees where erroring out which I removed from my list. Since I did not normalize my images for angle/jitter, my model would not perform well on those images (Turn Right Ahead as shown above). The background of the images would also affect the model like very dark, rainy or snowy backgrounds (Children Crossing). Some images that are blurred and blend with sky or background would be hard to classify using this model (No Vehicles).

# In[112]:

### Run the predictions here.
### Feel free to use as many code cells as needed.

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, save_path)
    test_accuracy = sess.run(accuracy_operation, feed_dict={x: X_new, y: y_new, keep_prob:1})

    print("Test Accuracy = {:.3f}".format(test_accuracy))
  


# ### Question 7
# 
# _Is your model able to perform equally well on captured pictures when compared to testing on the dataset? The simplest way to do this check the accuracy of the predictions. For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate._
# 
# _**NOTE:** You could check the accuracy manually by using `signnames.csv` (same directory). This file has a mapping from the class id (0-42) to the corresponding sign name. So, you could take the class id the model outputs, lookup the name in `signnames.csv` and see if it matches the sign from the image._
# 

# **Answer:**
# From the prediction I did to the candidate image, the testing accuracy is 60%, however, the model prediction accuracy on the training set was 97.7%. As a result, I believe my model did not perform well in the real world situation. 
# The possible reasons would be:
# 1) The angle of the image (Right Turn Ahead). Better Normalization of Jitter images would have helped
# 2) Blurred and images blended with sky or background for e.g  (Children Crossing) - Increasing the convolutional layers and decreasing the Filter shape might have helped
# 3) Signs not included in the training model would not have been classified correctly.

# In[113]:

### Visualize the softmax probabilities here.
### Feel free to use as many code cells as needed.

saver = tf.train.Saver()

softmax = tf.nn.softmax(logits)
top_3_op = tf.nn.top_k(softmax,3)

with tf.Session() as sess:
    saver.restore(sess, save_path)
    top_3 = sess.run(top_3_op, feed_dict={x: X_new, y: y_new, keep_prob:1})

plt.figure(figsize=(20,50))
for i in range(y_new.size):
    sub = plt.subplot(15,3,i+1)
    plt.imshow(X_new[i])
    sub.set_title("Prediction: %s, Certainty: %f" % (signnames[top_3[1][i][0]], top_3[0][i][0]))


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



