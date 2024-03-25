import numpy as np
import os
from torchvision import transforms
import torch
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from medclip import MedCLIPProcessor  # Assuming MedCLIPProcessor is defined elsewhere
from keras.preprocessing import image

# Set the random seed for reproducibility
np.random.seed(100)

processor = MedCLIPProcessor()

class JPEGToNumpy():
    def __call__(self, sample):
        jpeg_path, processor,model_type = sample
        if model_type == "medclip":
            img = Image.open(jpeg_path)
            inputs = processor(images=img)
        else:
            img = image.load_img(jpeg_path, target_size=(224, 224),
                             color_mode='rgb', interpolation='lanczos')
            inputs = np.asarray(img, dtype='uint8')
            inputs =  inputs / 255
        return inputs

class AiSeverity:
    def __init__(self, medical_type,model_type, device=None):
        self.medical_type = medical_type
        self.transforms = transforms.Compose([
            JPEGToNumpy()
        ])
        self.processor = processor
        self.device = device
        self.model_type = model_type

    def __call__(self, jpeg_path):
        
        inputs = self.transforms((jpeg_path, self.processor,self.model_type))
        return inputs

def process_folder(folder_path, ai_severity):
    subdirs = ["Positive", "Negative"]
    file_labels = {}
    file_samples = {}
    for subdir in subdirs:
        current_folder_path = os.path.join(folder_path, subdir)
        files = os.listdir(current_folder_path)
        label = 1 if subdir == "Positive" else 0
        for file_name in files:
            file_path = os.path.join(current_folder_path, file_name)
            if os.path.isfile(file_path):
                sample = ai_severity(file_path)
                file_samples[file_path] = sample
                file_labels[file_path] = label

    return file_samples, file_labels

def prepare_data_generators(samples, labels, batch_size,model_type):
    if model_type == "medclip":
        image_list = [(samples[key]["pixel_values"][0], labels[key]) for key in samples.keys()]
    else:
        image_list = [(samples[key], labels[key]) for key in samples.keys()]
    datagen = ImageDataGenerator(
        fill_mode="nearest",
        validation_split=0.20,
    )
    generator = datagen.flow(
        x=np.array([image for image, label in image_list]),
        y=np.array([label for image, label in image_list]),
        batch_size=batch_size,
        shuffle=True,
        seed=60
    )
    return len(image_list) // batch_size, generator

def create_loader(medical_type,batch_size,model_type):
    ai_severity = AiSeverity(medical_type, model_type,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Process folders
    train_samples, train_labels = process_folder(f'{medical_type}/Train/', ai_severity)
    validation_samples, validation_labels = process_folder(f'{medical_type}/Validation/', ai_severity)
    test_samples, test_labels = process_folder(f'{medical_type}/Test/', ai_severity)
    
    # Prepare data generators
    train_length,train_generator = prepare_data_generators(train_samples, train_labels, batch_size,model_type)
    validation_length, validation_generator = prepare_data_generators(validation_samples, validation_labels, batch_size,model_type)
    test_length, test_generator = prepare_data_generators(test_samples, test_labels, batch_size,model_type)
    
    return [train_generator, validation_generator, test_generator], [train_length, validation_length, test_length]