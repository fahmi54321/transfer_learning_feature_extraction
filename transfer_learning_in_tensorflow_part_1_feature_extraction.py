# Downloading nad becoming one with the data.

# https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip

# Get data (10% if 10 food class from food101) - https://www.kaggle.com/datasets/dansbecker/food-101
import zipfile
import wget

# Download the data

url = "https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip"
wget.download(url, 'pizza_steak.zip') 

# Unzip the download file
zip_ref = zipfile.ZipFile("10_food_classes_10_percent.zip")
zip_ref.extractall()
zip_ref.close()

# How many images in each folder?

import os

# Walk through 10 percent data directory and list number of files
for dirpath, dirnames, filenames in os.walk("10_food_classes_10_percent"):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
   
print(os.listdir("10_food_classes_10_percent"))

# Creating data loaders (preparing the data)

# Setup data inputs

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.applications import ResNet50V2, EfficientNetV2B0, EfficientNetV2B3, VGG19, EfficientNetB0

IMAGE_SHAPE = (224, 224)
BATCH_SIZE = 32

train_dir = "10_food_classes_10_percent/train/"
test_dir = "10_food_classes_10_percent/test/"

# train_datagen = ImageDataGenerator(rescale=1/255.)
# test_datagen = ImageDataGenerator(rescale=1/255.)

def get_preprocessing(keras_application):
    if keras_application == EfficientNetB0:
        from tensorflow.keras.applications.efficientnet import preprocess_input
    elif keras_application == ResNet50V2:
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
    elif keras_application == VGG19:
        from tensorflow.keras.applications.vgg19 import preprocess_input
    else:
        from tensorflow.keras.applications.imagenet_utils import preprocess_input
    return preprocess_input

preprocess_fn = get_preprocessing(EfficientNetB0)

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

print("Training images:")
train_data_10_percent = train_datagen.flow_from_directory(directory=train_dir,
                                               batch_size=BATCH_SIZE,
                                               target_size=IMAGE_SHAPE,
                                               class_mode="categorical")

print("Testing images:")
test_data = test_datagen.flow_from_directory(directory=test_dir,
                                               batch_size=BATCH_SIZE,
                                               target_size=IMAGE_SHAPE,
                                               class_mode="categorical")

# Setting up callbacks (things to run whilst our model trains)

# Create TensorBoard callback (functionized because we need to create a new one for each model)
import datetime
import tensorflow as tf

def create_tensorboard_callback(dir_name, experiment_name):
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback

# Import dependecies
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Flatten
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


def create_model(keras_application, num_classes = 10):

    base_model = keras_application(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')
    ])

    return model
    

# Create efficientnet model

train_data_10_percent.num_classes

efficientnet_model = create_model(keras_application=EfficientNetB0,num_classes=train_data_10_percent.num_classes)
resnet_model = create_model(keras_application=ResNet50V2,num_classes=train_data_10_percent.num_classes)


efficientnet_model.summary()
resnet_model.summary()

# Compile the model
efficientnet_model.compile(loss=CategoricalCrossentropy(),
              optimizer=Adam(),
              metrics=["accuracy"])

resnet_model.compile(loss=CategoricalCrossentropy(),
              optimizer=Adam(),
              metrics=["accuracy"])

history_efficientnet = efficientnet_model.fit(train_data_10_percent,
                      epochs=5,
                      steps_per_epoch=len(train_data_10_percent),
                      validation_data=test_data,
                      validation_steps=len(test_data),
                      callbacks = [create_tensorboard_callback(dir_name="tensorflow_hubs", experiment_name="efficient_net_v2_b0")])

history_resnet = resnet_model.fit(train_data_10_percent,
                      epochs=5,
                      steps_per_epoch=len(train_data_10_percent),
                      validation_data=test_data,
                      validation_steps=len(test_data),
                      callbacks = [create_tensorboard_callback(dir_name="tensorflow_hubs", experiment_name="resnet_model")])

import matplotlib.pyplot as plt

def plot_loss_curves(history):
    """
    Returns separate loss curves for training and validation metrices

    """
    
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    
    epochs = range(len(history.history["loss"])) # How many epochs did we run for?
    
    # Plot loss
    plt.plot(epochs, loss, label="training_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("loss")
    plt.xlabel("epochs")
    plt.legend()
    
    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label="training_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("accuracy")
    plt.xlabel("epochs")
    plt.legend()
    
    plt.show()

plot_loss_curves(history_efficientnet)

len(efficientnet_model.layers[0].weights)   

efficientnet_model.save('food_model_transfer_learning_feature_extraction_efficientnet_model.keras')
resnet_model.save('food_model_transfer_learning_feature_extraction_resnet_model.keras')












