import numpy as np
from tqdm import tqdm
import torch
import clip
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

class CLIPZeroShotClassifier:
    def __init__(self, medical_type, path=None):
        """
        Initializes the CLIPZeroShotClassifier with a specific medical type and computational device.
        
        :param medical_type: A string representing the medical classification task.
        :param device: The computational device ('cuda' or 'cpu') for computations. Automatically selected if None.
        :return: None.
        """
        self.medical_type = medical_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.configure()
        self.clip_model, self.preprocess = self.load_clip_model()
        if path is not None:
            self.clip_model.load_state_dict(torch.load(path))
    
    def configure(self):
        """
        Configures the system environment for optimal performance.
        :return: None. Prints the status of NVIDIA library configuration.
        """
        status = os.system('ldconfig /usr/lib64-nvidia')
        if status == 0:
            print("NVIDIA library configured successfully.")
        else:
            print("Error configuring NVIDIA library.")
        torch._dynamo.config.suppress_errors = True
        
    def load_clip_model(self):
        """
        Loads the CLIP model and preprocessing function into the specified device.
        :return: The CLIP model and its associated preprocessing function.
        """
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        return model, preprocess

    def zero_shot_classification(self, image_batch, categories):
        """
        Performs zero-shot classification using the CLIP model on a batch of images.
        :param image_batch: A tensor representing a batch of images.
        :param categories: A list of categories for classification.
        :return: The top probabilities and labels for the classification predictions.
        """
        text_inputs = torch.cat([clip.tokenize(f"a photo of {c} lungs.") for c in categories]).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_batch)
            text_features = self.clip_model.encode_text(text_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            top_probs, top_labels = similarity.topk(1, dim=-1)
        return top_probs, top_labels

    def evaluate(self, generators, steps, categories, visualize=True):
        """
        Evaluates the CLIP model using provided data loaders and computes classification metrics. 
        :param generators: A dictionary of data loaders for each dataset (e.g., 'Train', 'Validation', 'Test').
        :param steps: A dictionary specifying the number of batches to evaluate for each dataset.
        :param categories: A list of categories for classification.
        :return: Accuracy, precision, recall, AUC, classification report, and confusion matrix.
        """
        y_true, y_pred, y_score, y_input = [], [], [], []
        self.clip_model.eval()
        with torch.no_grad():
            for idx,(data_type, step) in enumerate(steps.items()):
                for _ in tqdm(range(step), desc=f'Evaluate {data_type}'):
                    inputs, labels = next(generators[idx])
                    ##inputs = self.preprocess(inputs)
                    inputs = torch.from_numpy(inputs).to(self.device)
                    labels = torch.from_numpy(labels).to(self.device).float().unsqueeze(1)
                    top_probs, top_labels = self.zero_shot_classification(inputs, categories)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(top_labels.cpu().numpy())
                    y_score.extend(top_probs.cpu().numpy())
                    y_input.extend(inputs.cpu().numpy())
                generators[idx].reset()
        if visualize:
            directory = os.path.join("results","visualization", self.experiment_type, self.medical_type, "clip", "images")
            filename = "prediction_images.pdf"
            filepath = os.path.join(directory, filename)
            os.makedirs(directory, exist_ok=True)
            sorted_indices = sorted(range(len(y_score)), key=lambda i: y_score[i], reverse=True)
            y_inputs_x = []
            y_score_x = []
            y_true_x = []
            y_pred_x = []
            count_11 = 0  # True 1, Predicted 1
            count_10 = 0  # True 1, Predicted 0
            count_01 = 0  # True 0, Predicted 1
            count_00 = 0  # True 0, Predicted 0
            max_samples_per_scenario = 3 
            for i in sorted_indices:
              true_label = y_true[i]
              predicted_label = y_pred[i]
              if true_label == 1 and predicted_label == 1 and count_11 < max_samples_per_scenario:
                  y_inputs_x.append(y_input[i])
                  y_score_x.append(y_score[i])
                  y_true_x.append(y_true[i])
                  y_pred_x.append(y_pred[i])
                  count_11 += 1
              elif true_label == 1 and predicted_label == 0 and count_10 < max_samples_per_scenario:
                  y_inputs_x.append(y_input[i])
                  y_score_x.append(y_score[i])
                  y_true_x.append(y_true[i])
                  y_pred_x.append(y_pred[i])
                  count_10 += 1
              elif true_label == 0 and predicted_label == 1 and count_01 < max_samples_per_scenario:
                  y_inputs_x.append(y_input[i])
                  y_score_x.append(y_score[i])
                  y_true_x.append(y_true[i])
                  y_pred_x.append(y_pred[i])
                  count_01 += 1
              elif true_label == 0 and predicted_label == 0 and count_00 < max_samples_per_scenario:
                  y_inputs_x.append(y_input[i])
                  y_score_x.append(y_score[i])
                  y_true_x.append(y_true[i])
                  y_pred_x.append(y_pred[i])
                  count_00 += 1
              if count_11 == max_samples_per_scenario and count_10 == max_samples_per_scenario and \
                count_01 == max_samples_per_scenario and count_00 == max_samples_per_scenario:
                  break
     
            fig, axs = plt.subplots(3,4, figsize=(15, 12))
            axs = axs.ravel()
            for i, (image, predicted, label) in enumerate(zip(y_inputs_x,y_pred_x,y_true_x)):
                axs[i].imshow(image.transpose(1, 2, 0))
                axs[i].set_title(f" True Label is {str(int(label[0]))} and Predicted Label is {str(predicted[0])}")
                axs[i].axis('off')
            for j in range(i + 1, len(axs)):
              axs[j].axis('off')
            plt.tight_layout()
            plt.savefig(filepath)
            print(f"Results saved to {filepath}")
            plt.close()
        acc, prec, rec, auc = accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred), roc_auc_score(y_true, y_score)
        cr, cm = classification_report(y_true, y_pred), confusion_matrix(y_true, y_pred)
        return acc, prec, rec, auc, cr, cm

    def save_results(self, acc, prec, rec, auc, cr, cm):
        """
        Saves the evaluation results to a file within a directory specific to the medical type and CLIP model.
        :param acc: The accuracy of the classification.
        :param prec: The precision of the classification.
        :param rec: The recall of the classification.
        :param auc: The AUC of the classification.
        :param cr: The classification report.
        :param cm: The confusion matrix.
        :return: None. Results are saved to a text file.
        """
        directory = f"results/{self.experiment_type}/{self.medical_type}/clip"
        filename = "classification_results.txt"
        filepath = os.path.join(directory, filename)
        os.makedirs(directory, exist_ok=True)
        with open(filepath, "w") as file:
            file.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nAUC: {auc:.4f}\n")
            file.write(f'Classification Report\n\n{cr}\n\nConfusion Matrix\n\n{np.array2string(cm)}')
        print(f"Results saved to {filepath}")

    def run(self, generators, steps, experiment_type, categories = ['normal', 'covid']):
        """
        Coordinates the process of zero-shot classification evaluation and result saving for the CLIP model.
        :param generators: A dictionary of data loaders for each dataset.
        :param steps: A dictionary specifying the number of batches to evaluate for each dataset.
        :param categories: A list of categories for classification.
        :return: None. Prints the evaluation metrics and saves the results.
        """
        self.experiment_type = experiment_type
        acc, prec, rec, auc, cr, cm = self.evaluate(generators, steps, categories)
        print(f"\nAccuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, AUC: {auc:.4f}")
        self.save_results(acc, prec, rec, auc, cr, cm)