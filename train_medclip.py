import data_loader.extract_data as extract_data
import data_loader.load_data as load_data
import visualize.visualize as visualize
from fine_tune.finetune_medclip import TrainMedClipClassifier
def run_finetune_clip(medical_type, model_type, batch_size):
    """
    Performs fine-tuning of the MedCLIP model on a given medical type and model type.
    :param medical_type: The type of medical data to classify ('ucsd', 'ori').
    :param model_type: The type of model to use for classification ('medclip', 'clip').
    :param batch_size: The batch size for data loading.
    """
    generators, lengths = load_data.create_loader(medical_type, batch_size, model_type)
    visualize.save_random_images_from_generators(generators, [medical_type, model_type, "fine_tune"], 2)
    if model_type == "medclip":
        classifier = TrainMedClipClassifier(medical_type)
        classifier.run(generators, lengths,1)
        return classifier
    else:
        print("Did not define a proper classifer!")

if __name__ == "__main__":
    extract_data.mount_and_process()
    batch_size = 256
    model_types = ['medclip']
    medical_types = ['ucsd', 'ori']
    ucsd_classifier = run_finetune_clip(medical_types[0], model_types[0], batch_size)
    ori_classifier = run_finetune_clip(medical_types[1], model_types[0], batch_size)
    