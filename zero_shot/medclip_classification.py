import numpy as np
from tqdm import tqdm
import torch
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel, PromptClassifier
from medclip.prompts import generate_covid_class_prompts, process_class_prompts, generate_rsna_class_prompts
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

class MedCLIPZeroShotClassifier:
    def __init__(self, medical_type, vision_model_cls=MedCLIPVisionModelViT,path=None):
        """
        Initializes the classifier with a specific medical type and model configuration.
        :param medical_type: A string representing the medical classification task.
        :param vision_model_cls: The vision model class from MedCLIP to use. Defaults to MedCLIPVisionModelViT.
        :param device: The device ('cuda' or 'cpu') for computation. If None, it auto-selects based on availability.
        :return: None.
        """
        self.medical_type = medical_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.configure()
        model, self.medclip_model = self.load_medclip_model(vision_model_cls)
        if path is not None:
            model.load_state_dict(torch.load(path))
            
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
        
    def load_medclip_model(self, vision_model_cls):
        """
        Loads the MedCLIP model based on the specified vision model class.
        :param vision_model_cls: The class of the vision model to load.
        :return: The PromptClassifier instance encapsulating the MedCLIP model.
        """
        model = MedCLIPModel(vision_cls=vision_model_cls)
        model.from_pretrained()
        model.to(self.device)
        clf = PromptClassifier(model, ensemble=False)
        clf.to(self.device)
        return model, clf


    def zero_shot_classification(self, image_batch, task, n):
        """
        Performs zero-shot classification on an image batch for the specified task using n prompts.
        :param image_batch: A batch of images to classify.
        :param task: The classification task ('rsna_task' or 'covid_task').
        :param n: The number of prompts to use for classification.
        :return: The top probabilities and labels for the classification predictions.
        """
        with torch.no_grad():
            task_type = generate_rsna_class_prompts(n=n) if task == "rsna_task" else generate_covid_class_prompts(n=n)
            input_dictionary = {'pixel_values': image_batch}
            cls_prompts = process_class_prompts(task_type)
            input_dictionary['prompt_inputs'] = cls_prompts
            output = self.medclip_model(**input_dictionary)['logits'].cpu().numpy()
            pred_score = torch.tensor(output.reshape(1, -1)[0]).sigmoid().numpy().flatten()
            pred_label = np.ones(len(pred_score))
            pred_label[pred_score<0.6] = 0
        return pred_score, pred_label

    def evaluate(self, generators, steps, task, n, visualize=False):
        """
        Evaluates the classifier performance on given datasets for a specified task and number of prompts.
        :param generators: A dictionary of data loaders for each dataset (e.g., 'Train', 'Validation', 'Test').
        :param steps: A dictionary specifying the number of evaluation steps for each dataset.
        :param task: The specific classification task to evaluate.
        :param n: The number of prompts to use for zero-shot classification.
        :return: Accuracy, precision, recall, AUC, classification report, and confusion matrix of the evaluation.
        """
        y_true, y_pred, y_score,y_input = [], [], [], []
        self.medclip_model.eval()
        with torch.no_grad():
            for idx,(data_type, step) in enumerate(steps.items()):
                for _ in tqdm(range(step), desc=f'Evaluate {data_type}'):
                    inputs, labels = next(generators[idx])
                    inputs = torch.from_numpy(inputs).to(self.device)
                    labels = torch.from_numpy(labels).to(self.device).float().unsqueeze(1)
                    top_probs, top_labels = self.zero_shot_classification(inputs, task, n)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(top_labels)
                    y_score.extend(top_probs)
                    y_input.extend(inputs.cpu().numpy())
                generators[idx].reset()
        if visualize:
            directory = os.path.join("results","visualization", self.experiment_type, self.medical_type, "medclip", "images")
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
            fig, axs = plt.subplots(3,4, figsize=(12, 8))
            axs = axs.ravel()
            for i, (image, predicted, label) in enumerate(zip(y_inputs_x,y_pred_x,y_true_x)):
                axs[i].imshow(image.transpose(1, 2, 0))
                axs[i].set_xticks([])  
                axs[i].set_yticks([])  
                if predicted == int(label[0]):
                    axs[i].patch.set_edgecolor('green')
                    axs[i].patch.set_linewidth(4)
                else:
                    axs[i].patch.set_edgecolor('red')
                    axs[i].patch.set_linewidth(4)
                axs[i].set_title(f"True Label = {str(int(label[0]))}", fontsize = 18)
            for j in range(i + 1, len(axs)):
              axs[j].axis('off')
            plt.tight_layout()
            plt.savefig(filepath)
            print(f"Results saved to {filepath}")
            plt.close()
        
        acc, prec, rec, auc = accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred), roc_auc_score(y_true, y_score)
        cr, cm = classification_report(y_true, y_pred), confusion_matrix(y_true, y_pred)
        return acc, prec, rec, auc, cr, cm

    def save_results(self, task, n_prompts, acc, prec, rec, auc, cr, cm):
        """
        Saves the evaluation results to a file.
        
        :param task: The classification task for which results are being saved.
        :param n_prompts: The number of prompts used for classification.
        :param acc: The accuracy of the classification.
        :param prec: The precision of the classification.
        :param rec: The recall of the classification.
        :param auc: The AUC of the classification.
        :param cr: The classification report.
        :param cm: The confusion matrix.
        :return: None. Results are saved to a file in the specified directory.
        """
        directory = f"results/{self.experiment_type}/{self.medical_type}/medclip"
        filename = "classification_results.txt"
        filepath = os.path.join(directory, filename)
        os.makedirs(directory, exist_ok=True)
        with open(filepath, "w") as file:
            file.write(f"Number of Prompts: {n_prompts}\nAccuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nAUC: {auc:.4f}\n")
            file.write(f'Classification Report\n\n{cr}\n\nConfusion Matrix\n\n{np.array2string(cm)}')
        print(f"Results saved to {filepath}")
        
    def run(self, generators, steps, experiment_type, tasks = ["covid_task"]):
        """
        Runs the zero-shot classification and evaluates performance across specified tasks.
        :param generators: A dictionary of data loaders for each dataset.
        :param steps: A dictionary specifying the number of evaluation steps for each dataset.
        :return: None. Prints and saves the evaluation results.
        """
        self.experiment_type = experiment_type
        for task in tasks:
            best_auc = 0
            best_metrics = None
            for n_prompts in range(1, 13):
                acc, prec, rec, auc, cr, cm = self.evaluate(generators, steps, task, n_prompts)
                print(f"\nAccuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, AUC: {auc:.4f}")
                if auc > best_auc:
                    best_auc = auc
                    best_metrics = (acc, prec, rec, auc, cr, cm, n_prompts)

            if best_metrics:
                acc, prec, rec, auc, cr, cm, n_prompts = best_metrics
                acc, prec, rec, auc, cr, cm = self.evaluate(generators, steps, task, n_prompts, visualize=True)
                print(f"Best AUC for {task} with {n_prompts} prompts: {auc:.4f}")
                self.save_results(task, n_prompts, acc, prec, rec, auc, cr, cm)