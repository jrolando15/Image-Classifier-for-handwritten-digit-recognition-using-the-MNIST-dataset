# Image-Classifier-for-handwritten-digit-recognition-using-the-MNIST-dataset


## Project Description
This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The MNIST dataset is a widely used benchmark dataset in the field of computer vision and machine learning, consisting of 60,000 training images and 10,000 test images of handwritten digits (0-9).

With this model, we obtained an accuracy of 0.98 on the test set.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Processing](#data-processing)
- [Model Architecture](#model-architecture)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Prediction](#prediction)
- [Contributing](#contributing)
- [License](#license)

## Installation
To run this project, you need to have Python installed along with the following libraries:
- TensorFlow
- NumPy

You can install the required libraries using the following command:
```bash
pip install tensorflow
```

## Usage 
1) Clone repository
```bash
git clone https://github.com/your_username/Image-Classifier-for-handwritten-digit-recognition-using-the-MNIST-dataset.git
cd Image-Classifier-for-handwritten-digit-recognition-using-the-MNIST-dataset
```

2) Run the Jupyter Notebook
```bash
jupyter notebook
```

## Project Structure
```bash
Image-Classifier-for-handwritten-digit-recognition-using-the-MNIST-dataset/
├── mnist_digit_classifier.ipynb  # Jupyter notebook with the code
├── README.md                     # Project README file
└── my_cnn_model.h5               # Saved model (after training)
```

## Data Preprocessing
The MNIST dataset is loaded using the mnist.load_data() function from TensorFlow's Keras datasets. The images are reshaped and normalized, and the labels are one-hot encoded.
```bash
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
```

## Model Architecture
The CNN model is defined using TensorFlow's Keras Sequential API. The model consists of the following layers:
Convolutional Layer (32 filters, 3x3 kernel)
Max Pooling Layer (2x2 pool size)
Convolutional Layer (64 filters, 3x3 kernel)
Max Pooling Layer (2x2 pool size)
Convolutional Layer (64 filters, 3x3 kernel)
Flatten Layer
Dense Layer (64 units, ReLU activation)
Dense Layer (10 units, Softmax activation)
```bash
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

## Model Training
The model is compiled with the Adam optimizer and categorical cross-entropy loss function. Data augmentation is applied to the training data using the ImageDataGenerator class. The model is trained for 9 epochs using the fit() function.
```bash
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
     rotation_range=10,
     width_shift_range=0.1,
     height_shift_range=0.1,
     zoom_range=0.1,
     horizontal_flip=True
)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

train_generator = datagen.flow(train_images, train_labels, batch_size=64)
model.fit(train_generator, epochs=9, validation_data=(test_images, test_labels))
```

## Model Evaluation
The trained model is evaluated on the test set, and the test accuracy is printed.

```bash
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'test accuracy: {test_acc}')
```

## Prediction
The trained model can be used to make predictions on new images. The predict() function is used to obtain the predicted class probabilities for the first 5 test images.
```bash
predictions = model.predict(test_images[:5])
```

## License
This README file provides an overview of the project, including its description, installation instructions, usage, project structure, data processing, model architecture, training, evaluation, and prediction. It also includes sections for contributing and licensing, which are important for open-source projects.
