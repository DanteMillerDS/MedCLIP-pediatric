import shutil
from google.colab import drive
import numpy as np
import os
from torchvision import transforms
import torch
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import argparse
from medclip import MedCLIPProcessor
np.random.seed(100)

processor = MedCLIPProcessor()

class JPEGToNumpy():
    def __call__(self, sample):
        image = Image.open(sample[0]['jpeg_path'])
        processor = sample[1]
        inputs = processor(images=image)
        sample[0]['image'] = inputs
        return sample

class AiSeverity:
    def __init__(self, medical_type, device=None):
        self.medical_type = medical_type
        self.transforms = transforms.Compose([
            JPEGToNumpy()
        ])

    def __call__(self, jpeg_path):
        sample = {'jpeg_path': jpeg_path}
        sample = self.transforms([sample,processor])
        return sample[0]['image']
    
def process_folder(folder_path, ai_severity):
    
    file_labels = {}
    file_samples = {}

    files = os.listdir(folder_path)
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        label = 1 if 'Positive' in folder_path else 0
        file_labels[file_path] = label

    index = 0
    
    items_list = list(file_labels.items())

# Shuffle the list of tuples using random.shuffle()
    np.random.shuffle(items_list)

    # Create a new dictionary from the shuffled list of tuples
    file_labels = dict(items_list)
    for file_path, label in file_labels.items():
        sample = ai_severity(file_path)
        print(f'{file_path}: label is {label}')
        file_samples[file_path] = sample
        index += 1
        print(f"Index: {index}")

    return file_samples, file_labels

def main(medical_type):
    # Initialize AiSeverity with medical_type
    ai_severity = AiSeverity(medical_type, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Process folders
    train_samples, train_labels = process_folder(f'{medical_type}/Train/Positive/', ai_severity)
    validation_samples, validation_labels = process_folder(f'{medical_type}/Validation/Positive/', ai_severity)
    test_samples, test_labels = process_folder(f'{medical_type}/Test/Positive/', ai_severity)
    
    BATCH_SIZE_TRAINING = 32
BATCH_SIZE_VALIDATION = 32
BATCH_SIZE_TESTING = 1

train_datagen = ImageDataGenerator(
fill_mode="nearest",
    validation_split=0.20,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,

)

test_datagen = ImageDataGenerator(
fill_mode="nearest",
    validation_split=0.20,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
)

train_image_list = [(train_samples[key]["pixel_values"][0], train_labels[key]) for key in train_samples.keys()]
validation_image_list = [(validation_samples[key]["pixel_values"][0], validation_labels[key]) for key in validation_samples.keys()]
test_image_list = [(test_samples[key]["pixel_values"][0], test_labels[key]) for key in test_samples.keys()]

# Split the image_list into training and validation sets
train_generator_train = train_datagen.flow(
    x=np.array([image for image, label in train_image_list]),
    y=[label for image, label in train_image_list],
    batch_size=BATCH_SIZE_TRAINING,
    shuffle=True,
    seed=60
)

valid_generator_train = train_datagen.flow(
    x=np.array([image for image, label in validation_image_list]),
    y=[label for image, label in validation_image_list],
    batch_size=BATCH_SIZE_VALIDATION,
    shuffle=True,
    seed=60
)

test_generator_train = test_datagen.flow(
    x=np.array([image for image, label in test_image_list]),
    y=[label for image, label in test_image_list],
    batch_size=BATCH_SIZE_TESTING,
    shuffle=False,
    seed=60
)
        return train_generator_train, valid_generator_train, test_generator_train
    
    

