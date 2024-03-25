import numpy as np
import matplotlib.pyplot as plt
import os

def save_random_images_from_generators(train_generator, valid_generator, test_generator, info, num_images=2):
    medical_type,model_type = info
    directory = os.path.join(medical_type, model_type, "images")
    filename = "cxr_images.png"
    filepath = os.path.join(directory, filename)
    os.makedirs(directory, exist_ok=True)
    train_images, _ = next(train_generator)
    valid_images, _ = next(valid_generator)
    test_images, _ = next(test_generator)
    plt.figure(figsize=(12, 8))
    for i in range(num_images):
        idx = np.random.randint(0, train_images.shape[0])
        plt.subplot(3, num_images, i + 1)
        plt.imshow(train_images[idx])
        plt.title("Train")
        plt.subplot(3, num_images, num_images + i + 1)
        plt.imshow(valid_images[idx])
        plt.title("Validation")
        plt.subplot(3, num_images, 2 * num_images + i + 1)
        plt.imshow(test_images[idx])
        plt.title("Test")
    plt.tight_layout()
    plt.savefig(filepath)
    print(f"Images saved to {filepath}")
    plt.close()
    train_generator.reset()
    valid_generator.reset()
    test_generator.reset()
    
