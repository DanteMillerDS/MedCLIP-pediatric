import numpy as np
import os
from PIL import Image
from torchvision import transforms
import torch
from keras.preprocessing import image
from medclip import MedCLIPProcessor
np.random.seed(100)

class ImagePreprocessor:
    """
    A class to preprocess images based on the model type specified.
    It supports preprocessing for both 'medclip' and other generic models
    by adjusting the image format accordingly.
    """
    def __init__(self, processor, model_type):
        self.processor = processor
        self.model_type = model_type

    def __call__(self, jpeg_path):
        """
        Loads and preprocesses an image based on the model type.
        :param jpeg_path: Path to the JPEG image file.
        :return: Preprocessed image data.
        """
        if self.model_type == "medclip":
            img = Image.open(jpeg_path)
            inputs = self.processor(images=img)
        else:
            img = image.load_img(jpeg_path, target_size=(224, 224),
                                 color_mode='rgb', interpolation='lanczos')
            inputs = np.asarray(img, dtype='uint8') / 255
        return inputs

class AiSeverity:
    """
    A class to manage the severity analysis AI models for medical images,
    including initializing the model, device setup, and image preprocessing.
    """
    def __init__(self, medical_type, model_type, device=None):
        self.medical_type = medical_type
        self.model_type = model_type
        self.processor = MedCLIPProcessor()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = ImagePreprocessor(processor=self.processor, model_type=self.model_type)

    def process_image(self, jpeg_path):
        """
        Processes an image through the preprocessor.
        :param jpeg_path: Path to the image file.
        :return: Processed image data.
        """
        return self.preprocessor(jpeg_path)

def process_folder(folder_path, ai_severity):
    """
    Processes all images in a given folder, labeling them based on their subdirectory.
    :param folder_path: Path to the base folder containing subdirectories for classification.
    :param ai_severity: An instance of AiSeverity for processing images.
    :return: Two dictionaries with image paths as keys and processed images or labels as values.
    """
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
                sample = ai_severity.process_image(file_path)
                file_samples[file_path] = sample
                file_labels[file_path] = label
    return file_samples, file_labels

def prepare_data_generators(samples, labels, batch_size, model_type):
    """
    Prepares data generators for training, validation, and testing datasets.
    :param samples: A dictionary of image paths and their processed data.
    :param labels: A dictionary of image paths and their labels.
    :param batch_size: Batch size for the data generator.
    :param model_type: The model type to tailor the data preparation.
    :return: The length of data and the data generator.
    """
    image_list = [(sample if model_type != "medclip" else sample["pixel_values"][0], labels[key]) 
                  for key, sample in samples.items()]
    datagen = image.ImageDataGenerator(validation_split=0.20)
    generator = datagen.flow(
        x=np.array([image for image, label in image_list]),
        y=np.array([label for image, label in image_list]),
        batch_size=batch_size,
        shuffle=True,
        seed=60
    )
    return len(image_list) // batch_size, generator

def create_loader(medical_type, batch_size, model_type):
    """
    Initializes the AISeverity model, processes image folders, and prepares data generators.
    :param medical_type: The medical condition or type to analyze.
    :param batch_size: The batch size for the data loaders.
    :param model_type: The model type for specific preprocessing needs.
    :return: Lists of data generators and their corresponding lengths.
    """
    ai_severity = AiSeverity(medical_type, model_type)
    train_samples, train_labels = process_folder(f'{medical_type}/Train', ai_severity)
    validation_samples, validation_labels = process_folder(f'{medical_type}/Validation', ai_severity)
    test_samples, test_labels = process_folder(f'{medical_type}/Test', ai_severity)
    train_length, train_generator = prepare_data_generators(train_samples, train_labels, batch_size, model_type)
    validation_length, validation_generator = prepare_data_generators(validation_samples, validation_labels, batch_size, model_type)
    test_length, test_generator = prepare_data_generators(test_samples, test_labels, batch_size, model_type)
    lengths = [train_length, validation_length, test_length]
    dataset_types = ['Train', 'Validation', 'Test']
    steps = dict(zip(dataset_types, lengths))
    return [train_generator, validation_generator, test_generator], steps