import data_loader.extract_data as extract_data
import data_loader.load_data as load_data
import visualize.visualize as visualize
from zero_shot.clip_classification import CLIPZeroShotClassifier
from zero_shot.medclip_classification import MedCLIPZeroShotClassifier
def run_classification_process(medical_type, model_type, batch_size):
    """
    Handles the process of running zero-shot classification for a given model type and medical type.
    :param medical_type: The type of medical data to classify ('ucsd', 'ori').
    :param model_type: The type of model to use for classification ('medclip', 'clip').
    :param batch_size: The batch size for data loading.
    """
    generators, lengths = load_data.create_loader(medical_type, batch_size, model_type)
    visualize.save_random_images_from_generators(generators, [medical_type, model_type, "zero_shot"], 2)
    if model_type == "clip":
        classifier = CLIPZeroShotClassifier(medical_type)
        classifier.run(generators, lengths)
    elif model_type == "medclip":
        classifier = MedCLIPZeroShotClassifier(medical_type)
        classifier.run(generators, lengths)
    else:
        print("Did not define a proper classifer!")

if __name__ == "__main__":
    extract_data.mount_and_process()
    batch_size = 256
    model_types = ['medclip', 'clip']
    medical_types = ['ucsd', 'ori']
    for medical_type in medical_types:
        for model_type in model_types:
            run_classification_process(medical_type, model_type, batch_size)
