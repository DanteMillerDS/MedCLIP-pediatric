import numpy as np
import os
from torchvision import transforms
import torch
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from medclip import MedCLIPProcessor  # Assuming MedCLIPProcessor is defined elsewhere

# Set the random seed for reproducibility
np.random.seed(100)

processor = MedCLIPProcessor()

class JPEGToNumpy():
    def __call__(self, sample):
        image, processor = sample
        inputs = processor(images=image)
        return inputs

class AiSeverity:
    def __init__(self, medical_type, device=None):
        self.medical_type = medical_type
        self.transforms = transforms.Compose([
            JPEGToNumpy()
        ])
        self.processor = processor
        self.device = device

    def __call__(self, jpeg_path):
        image = Image.open(jpeg_path)
        inputs = self.transforms((image, self.processor))
        return inputs

def process_folder(folder_path, ai_severity):
    file_labels = {}
    file_samples = {}
    files = os.listdir(folder_path)
    
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        label = 1 if 'Positive' in folder_path else 0
        sample = ai_severity(file_path)
        file_samples[file_path] = sample
        file_labels[file_path] = label

    return file_samples, file_labels

def prepare_data_generators(samples, labels, batch_size):
    image_list = [(samples[key]["pixel_values"][0], labels[key]) for key in samples.keys()]
    datagen = ImageDataGenerator(
        fill_mode="nearest",
        validation_split=0.20,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
    )
    generator = datagen.flow(
        x=np.array([image for image, label in image_list]),
        y=np.array([label for image, label in image_list]),
        batch_size=batch_size,
        shuffle=True,
        seed=60
    )
    return len(image_list), generator

def create_loader(medical_type,batch_size):
    ai_severity = AiSeverity(medical_type, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Process folders
    train_samples, train_labels = process_folder(f'{medical_type}/Train/Positive/', ai_severity)
    validation_samples, validation_labels = process_folder(f'{medical_type}/Validation/Positive/', ai_severity)
    test_samples, test_labels = process_folder(f'{medical_type}/Test/Positive/', ai_severity)
    
    # Prepare data generators
    train_length,train_generator = prepare_data_generators(train_samples, train_labels, batch_size=batch_size)
    validation_length, validation_generator = prepare_data_generators(validation_samples, validation_labels, batch_size=batch_size)
    test_length, test_generator = prepare_data_generators(test_samples, test_labels, batch_size=1)
    
    return train_generator, validation_generator, test_generator, train_length, validation_length, test_length
