import numpy as np
import matplotlib.pyplot as plt
import os

def save_random_images_from_generators(generators, info, num_images=2):
    """
    Saves random images from data generators into a combined image file.
    :param generators: A list of data generators from which to draw images.
    :param info: A tuple containing (medical_type, model_type) to define the directory for saving images.
    :param num_images: The number of random images to save from each generator. Default is 2.
    :return: None. Images are saved to a file named 'cxr_images.png' within a directory path constructed from the 'info' tuple.
    """
    medical_type, model_type = info
    directory = os.path.join(medical_type, model_type, "images")
    filename = "cxr_images.png"
    filepath = os.path.join(directory, filename)
    os.makedirs(directory, exist_ok=True)
    plt.figure(figsize=(12, 8))
    for gen, title in zip(generators, ["Train", "Validation", "Test"]):
        images, _ = next(gen) 
        for i in range(num_images):
            idx = np.random.randint(0, images.shape[0])
            image = images[idx]
            if image.shape[0] < image.shape[-1]:
                image = image.transpose(1, 2, 0)
            plt.subplot(3, num_images, i + 1 + (["Train", "Validation", "Test"].index(title) * num_images))
            plt.imshow(image)
            plt.title(title)
        gen.reset()
    plt.tight_layout()
    plt.savefig(filepath)
    print(f"Results saved to {filepath}")
    plt.close()

    
    

