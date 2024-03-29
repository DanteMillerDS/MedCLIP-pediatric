import numpy as np
from tqdm import tqdm
from torch import nn
import torch.optim as optim
import torch
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel, PromptClassifier
from medclip.prompts import generate_covid_class_prompts, process_class_prompts, generate_rsna_class_prompts
import clip
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import os
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

class TrainMedClipClassifier:
    def __init__(self, medical_type, epochs=25):
        """
        Initializes the TrainMedClipClassifier with a specific medical type and computational device.
        """
        self.medical_type = medical_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.configure()
        self.medclip_model, self.clf = self.load_medclip_model(MedCLIPVisionModelViT)
        self.wd = 0.1 if self.medical_type == "ucsd" else 1e-4
        self.optimizer = optim.Adam(self.medclip_model.parameters(), lr=1e-5,weight_decay =self.wd)
        self.epochs = epochs
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()
        self.metric_history  = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': [],
            'train_auc': [],
            'val_auc': [],
        }
        self.early_stopping_patience = 5
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        self.early_stop = False
        self.max_grad_norm =1

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
  
    def convert_models_to_fp32(self, model):
        """
        Converts model parameters and gradients to float32 precision. This is necessary for compatibility with certain optimizers or hardware.
        :params model: The model to convert to float32 precision.
        :return: None. Converts the model parameters and gradients in-place.
        """   
        for p in model.parameters():
            if p.grad is not None:
                p.data = p.data.float()
                p.grad.data = p.grad.data.float()

    def load_medclip_model(self, vision_model_cls):
        """
        Loads the MedCLIP model based on the specified vision model class.
        :param vision_model_cls: The class of the vision model to load.
        :return: The PromptClassifier instance encapsulating the MedCLIP model.
        """
        model = MedCLIPModel(vision_cls=vision_model_cls)
        model.from_pretrained()
        model.to(self.device)
        clf = PromptClassifier(model)
        clf.to(self.device)
        return model, clf

    def zero_shot_classification(self, image_batch):
        """
         Performs zero-shot classification using the MedCLIP model on a batch of images.
        :param image_batch: A tensor representing a batch of images.
        :param categories: A list of categories for classification.
        :return: The top probabilities and labels for the classification predictions.
        """
        with torch.no_grad():
            texts = {"COVID": [f"a photo of covid lungs."]}
            input_dictionary = {'pixel_values': image_batch}
            cls_prompts = process_class_prompts(texts)
            input_dictionary['prompt_inputs'] = cls_prompts
            output = self.clf(**input_dictionary)['logits'].cpu().numpy()
            pred_score = torch.tensor(output.reshape(1, -1)[0]).sigmoid().numpy().flatten()
            pred_label = np.ones(len(pred_score))
            pred_label[pred_score<0.6] = 0
        return pred_score, pred_label

    def evaluate(self, generators, steps ):
        """
        Evaluates the classifier performance on given datasets for a specified task and number of prompts.
        :param generators: A dictionary of data loaders for each dataset (e.g., 'Train', 'Validation', 'Test').
        :param steps: A dictionary specifying the number of evaluation steps for each dataset.
        :param task: The specific classification task to evaluate.
        :param n: The number of prompts to use for zero-shot classification.
        :return: Accuracy, precision, recall, AUC, classification report, and confusion matrix of the evaluation.
        """
        y_true, y_pred, y_score = [], [], []
        with torch.no_grad() and autocast():
            for idx,(data_type, step) in enumerate(steps.items()):
                for _ in tqdm(range(step), desc=f'Evaluate {data_type}'):
                    inputs, labels = next(generators[idx])
                    inputs = torch.from_numpy(inputs).to(self.device)
                    labels = torch.from_numpy(labels).to(self.device).float().unsqueeze(1)
                    top_probs, top_labels = self.zero_shot_classification(inputs)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(top_labels)
                    y_score.extend(top_probs)
                generators[idx].reset()
   
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
        directory = f"results/finetune/{self.medical_type}/medclip"
        filename = "classification_results.txt"
        filepath = os.path.join(directory, filename)
        os.makedirs(directory, exist_ok=True)
        with open(filepath, "w") as file:
            file.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nAUC: {auc:.4f}\n")
            file.write(f'Classification Report\n\n{cr}\n\nConfusion Matrix\n\n{np.array2string(cm)}')
        print(f"Results saved to {filepath}")
       
    def train_validate(self, train_loader, validation_loader, steps, categories):
        """
        Coordinates the training and validation of the MedCLIP model for a specified number of epochs.
        param train_loader: The data loader for the training dataset.
        param validation_loader: The data loader for the validation dataset.
        param steps: A dictionary specifying the number of batches to train and validate for each dataset.
        param categories: A list of categories for classification.
        """
        model_save_path = f'results/finetune/{self.medical_type}/medclip/best_model.pth'
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.epochs):
            self.medclip_model.train()
            train_losses = []
            for step in tqdm(range(steps["Train"]), desc=f'Epoch {epoch+1}/{self.epochs}, Train'):
                inputs, labels = next(train_loader)
                inputs = torch.from_numpy(inputs).to(self.device)
                labels = torch.from_numpy(labels).to(self.device).float().unsqueeze(1)
                
                texts = {"COVID": [f"a photo of {categories[int(labels[i].item())]} lungs." for i in range(len(labels))]}
               
                cls_prompts = process_class_prompts(texts)
                self.optimizer.zero_grad()
                input_dictionary = {'pixel_values': inputs}
                input_dictionary['prompt_inputs'] = cls_prompts
                with autocast():
                  loss_value = self.medclip_model(input_ids=input_dictionary["prompt_inputs"]["COVID"]["input_ids"],
                                              pixel_values=input_dictionary["pixel_values"],
                                              attention_mask=input_dictionary["prompt_inputs"]["COVID"]["attention_mask"],
                                              return_loss = True)['loss_value']
                  scaler.scale(loss_value).backward()
                  scaler.unscale_(self.optimizer)
                  torch.nn.utils.clip_grad_norm_(self.medclip_model.parameters(), self.max_grad_norm)
                  scaler.step(self.optimizer)
                  scaler.update()
                  
                train_losses.append(loss_value.item())
            avg_train_loss = np.mean(train_losses)
            self.medclip_model.eval()
            validation_losses = []
            for step in tqdm(range(steps["Validation"]), desc=f'Epoch {epoch+1}/{self.epochs}, Validation'):
                inputs, labels = next(validation_loader)
                inputs = torch.from_numpy(inputs).to(self.device)
                labels = torch.from_numpy(labels).to(self.device).float().unsqueeze(1)
                
                texts = {"COVID": [f"a photo of {categories[int(labels[i].item())]} lungs." for i in range(len(labels))]}
               
                cls_prompts = process_class_prompts(texts)
                self.optimizer.zero_grad()
                input_dictionary = {'pixel_values': inputs}
                input_dictionary['prompt_inputs'] = cls_prompts
                with autocast():
                  loss_value = self.medclip_model(input_ids=input_dictionary["prompt_inputs"]["COVID"]["input_ids"],
                                              pixel_values=input_dictionary["pixel_values"],
                                              attention_mask=input_dictionary["prompt_inputs"]["COVID"]["attention_mask"],
                                              return_loss = True)['loss_value']
                                  
                validation_losses.append(loss_value.item())
            avg_validation_loss = np.mean(validation_losses)
            train_acc, train_prec, train_rec, train_auc, _, _ = self.evaluate([train_loader], {"Train":steps["Train"]})
            val_acc, val_prec, val_rec, val_auc, _, _ = self.evaluate([validation_loader], {"Validation":steps["Validation"]})
            self.metric_history['train_loss'].append(avg_train_loss)
            self.metric_history['val_loss'].append(avg_validation_loss)
            self.metric_history['train_accuracy'].append(train_acc)
            self.metric_history['val_accuracy'].append(val_acc)
            self.metric_history['train_precision'].append(train_prec)
            self.metric_history['val_precision'].append(val_prec)
            self.metric_history['train_recall'].append(train_rec)
            self.metric_history['val_recall'].append(val_rec)
            self.metric_history['train_auc'].append(train_auc)
            self.metric_history['val_auc'].append(val_auc)

            epochs_range = range(1, epoch + 2)
            for i, (metric_name) in enumerate(['loss', 'accuracy', 'precision', 'recall', 'auc'], 1):
                plt.figure(figsize=(10, 6))
                plt.plot(epochs_range, self.metric_history[f'train_{metric_name}'], label=f'Train {metric_name.capitalize()}')
                plt.plot(epochs_range, self.metric_history[f'val_{metric_name}'], label=f'Validation {metric_name.capitalize()}', linestyle='--')
                plt.legend(loc='best')
                plt.title(metric_name.capitalize())
                plt.tight_layout()
                plt.savefig(f'results/finetune/{self.medical_type}/medclip/metrics_{metric_name}_epoch.png')
                plt.close()

            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"Train - Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, AUC: {train_auc:.4f}")
            print(f"Val - Loss: {avg_validation_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, AUC: {val_auc:.4f}")
            if avg_validation_loss < self.best_val_loss:
                self.best_val_loss = avg_validation_loss
                torch.save(self.medclip_model.state_dict(), model_save_path)
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter == self.early_stopping_patience:
                    self.early_stop = True
                    print("Early stopping triggered.")
                    break
        self.medclip_model.load_state_dict(torch.load(model_save_path))

    def run(self, generators, steps, categories = ['normal', 'covid']):
        """
        Coordinates the process of zero-shot classification evaluation and result saving for the CLIP model.
        :param generators: A dictionary of data loaders for each dataset.
        :param steps: A dictionary specifying the number of batches to evaluate for each dataset.
        :param categories: A list of categories for classification.
        :return: None. Prints the evaluation metrics and saves the results.
        """
        self.train_validate(generators[0],generators[1],steps,categories)
        acc, prec, rec, auc, cr, cm = self.evaluate([generators[2]], {"Test":steps["Test"]})
        print(f"\nAccuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, AUC: {auc:.4f}")
        self.save_results(acc, prec, rec, auc, cr, cm)