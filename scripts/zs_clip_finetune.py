import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import data_loader.extract_data as extract_data
import data_loader.load_data as load_data
import visualize.visualize as visualize
from zero_shot.clip_classification import CLIPZeroShotClassifier
def run_classification_process_clip(medical_type, model_type, batch_size, path):
    """
    Handles the process of running zero-shot classification for clip and a given medical type.
    :param medical_type: The type of medical data to classify ('ucsd', 'ori').
    :param model_type: The type of model to use for classification ('medclip', 'clip').
    :param batch_size: The batch size for data loading.
    """
    generators, lengths = load_data.create_loader(medical_type, batch_size, model_type)
    visualize.save_random_images_from_generators(generators, [medical_type, model_type, "zero_shot_finetune"], 2)
    if model_type == "clip":
        classifier = CLIPZeroShotClassifier(medical_type, path = path)
        classifier.run(generators, lengths, "zs_clip_finetune")
    else:
        print("Did not define a proper classifer!")

if __name__ == "__main__":
    extract_data.mount_and_process()
    batch_size = 256
    model_types = ['clip']
    medical_types = ['ucsd','ori']
    paths = ["results/t_pretrained/ori/clip/best_model.pth","results/t_pretrained/ucsd/clip/best_model.pth"]
    run_classification_process_clip(medical_types[0], model_types[0], batch_size, paths[0])
    run_classification_process_clip(medical_types[1], model_types[0], batch_size,paths[1])
