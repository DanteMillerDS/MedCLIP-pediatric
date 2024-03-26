import numpy as np
import matplotlib.pyplot as plt
import os

def create_save_directory(info):
    """
    Creates a directory for saving images if it does not already exist.
    :param info: A tuple containing (medical_type, model_type) to define the directory.
    :return: The filepath where the images will be saved.
    """
    medical_type, model_type = info
    directory = os.path.join("results","visualization",medical_type, model_type, "images")
    filename = "cxr_images.png"
    filepath = os.path.join(directory, filename)
    os.makedirs(directory, exist_ok=True)
    return filepath

def select_random_images(generator, num_images=2):
    """
    Selects a specified number of random images from a data generator.
    :param generator: The data generator from which to draw images.
    :param num_images: The number of random images to select. Default is 2.
    :return: A list of randomly selected images.
    """
    images, _ = next(generator)
    selected_images = []
    for _ in range(num_images):
        idx = np.random.randint(0, images.shape[0])
        image = images[idx]
        if image.shape[0] < image.shape[-1]:
            image = image.transpose(1, 2, 0)
        selected_images.append(image)
    generator.reset()
    return selected_images

def plot_images(images, titles, filepath):
    """
    Plots a list of images and saves them to a file.
    :param images: A list of lists of images to plot.
    :param titles: The titles for each subplot.
    :param filepath: The filepath where the plot will be saved.
    :return: None.
    """
    plt.figure(figsize=(12, 8))
    for i, image_batch in enumerate(images):
        for j, image in enumerate(image_batch):
            plt.subplot(3, len(image_batch), j + 1 + (i * len(image_batch)))
            plt.imshow(image)
            plt.title(titles[i])
    plt.tight_layout()
    plt.savefig(filepath)
    print(f"Results saved to {filepath}")
    plt.close()

def save_random_images_from_generators(generators, info, num_images=2):
    """
    Saves random images from data generators into a combined image file.
    :param generators: A list of data generators from which to draw images.
    :param info: A tuple containing (medical_type, model_type) to define the directory for saving images.
    :param num_images: The number of random images to save from each generator. Default is 2.
    :return: None.
    """
    filepath = create_save_directory(info)
    all_images = []
    for gen in generators:
        selected_images = select_random_images(gen, num_images)
        all_images.append(selected_images)
        gen.reset()
    plot_images(all_images, ["Train", "Validation", "Test"], filepath)
