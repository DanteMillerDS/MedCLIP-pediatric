import numpy as np
import matplotlib.pyplot as plt
import os


def save_random_images_from_generators(train_generator, valid_generator, test_generator, info, num_images=2):
    medical_type,model_type = info
    directory = os.path.join(medical_type, model_type, "images")
    filename = "cxr_images.png"
    filepath = os.path.join(directory, filename)
    os.makedirs(directory, exist_ok=True)
    plt.figure(figsize=(12, 8))
    for gen, title in zip([train_generator, valid_generator, test_generator], ["Train", "Validation", "Test"]):
        images, _ = next(gen)  # Fetch a batch of images and labels
        for i in range(num_images):
            idx = np.random.randint(0, images.shape[0])  # Select a random index
            image = images[idx]

            if image.shape[0] < image.shape[-1]:  
                image = image.transpose(1, 2, 0)  
            plt.subplot(3, num_images, i + 1 + (["Train", "Validation", "Test"].index(title) * num_images))
            plt.imshow(image)
            plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    print(f"Results saved to {filepath}")
    plt.close()
    train_generator.reset()
    valid_generator.reset()
    test_generator.reset()
    
    
    

