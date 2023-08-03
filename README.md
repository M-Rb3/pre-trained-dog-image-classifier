# Pre-trained Dog Image Classifier

![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Description

Pre-trained Dog Image Classifier is a Python-based deep learning project that uses a pre-trained convolutional neural network (CNN) to classify dog images into different breeds. The model is built using PyTorch and leverages transfer learning to achieve high accuracy even with a limited dataset.

## Features

- Pre-trained CNN for dog breed classification
- Supports classification of custom dog images
- Easy-to-use command-line interface
- Detailed results and confidence scores

## Requirements

- Python 3.9 or higher
- PyTorch (torch) library
- TorchVision (torchvision) library
- NumPy (numpy) library

Install the required dependencies using the following command:

pip install torch torchvision numpy

markdown
Copy code

## Usage

1. Clone the repository:

      git clone https://github.com/M-Rb3/pre-trained-dog-image-classifier.git
      cd pre-trained-dog-image-classifier

      less
      Copy code

2. Download the pre-trained model weights:

      Download the model weights file `dog_classifier_weights.pth` from [Google Drive](https://drive.google.com/file/d/xyz1234567890/view?usp=sharing) and place it in the project root directory.

3. Classify a dog image:

      python classify_dog.py path/to/your/dog_image.jpg
      
      csharp
      Copy code

      Replace `path/to/your/dog_image.jpg` with the path to the dog image you want to classify.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The pre-trained model used in this project is based on [VGG16](https://arxiv.org/abs/1409.1556) architecture with transfer learning.
- The dog breed dataset used for training and validation is a subset of the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/).
