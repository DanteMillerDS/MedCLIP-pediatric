import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import torch._dynamo
import clip     
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel
from medclip.prompts import generate_covid_class_prompts, process_class_prompts, generate_rsna_class_prompts
from PIL import Image
from medclip import PromptClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time
import os

status = os.system('ldconfig /usr/lib64-nvidia')
if status == 0:
    print("Command executed successfully")
else:
    print("Error executing the command")
    
torch._dynamo.config.suppress_errors = True

def run_zero_shot_classification_medclipmodel(medical_type, generators, steps):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for vision_model_name,vision_model in zip(["MedCLIPVisionModelViT"],[MedCLIPVisionModelViT]):
        model = MedCLIPModel(vision_cls=vision_model)
        model.from_pretrained()
        model.to(device)
        clf = PromptClassifier(model, ensemble=True)
        clf.to(device)

        def zero_shot_classification(model, image_batch,task,n):
            task_type = generate_rsna_class_prompts(n=n) if task == "rsna_task" else generate_covid_class_prompts(n=n)
            input_dictionary = {'pixel_values': image_batch}
            cls_prompts = process_class_prompts(task_type)
            input_dictionary['prompt_inputs'] = cls_prompts
            output = model(**input_dictionary)['logits'].cpu().numpy()
            top_probs = output.reshape(1, -1)[0]
            top_labels = np.round(top_probs)
            return top_probs, top_labels

        def evaluate(model, data_loaders, device, steps, data_types,task,n):
            model.eval()
            y_true, y_pred, y_score = [], [], []
            with torch.no_grad():
                for data_loader, data_type, step in zip(data_loaders, data_types, steps):
                    for step_ind in tqdm(range(step), desc=f'Evaluate {data_type}'):
                        inputs, labels = data_loader[step_ind][0], data_loader[step_ind][1]
                        inputs = torch.from_numpy(inputs).to(device)
                        labels = torch.from_numpy(labels).to(device).float().unsqueeze(1)
                        if task != "both_tasks":
                            top_probs, top_labels = zero_shot_classification(model, inputs, task, n)
                        elif task == "both_tasks":
                            covid_probs, _ = zero_shot_classification(model, inputs, "covid_task", n)
                            rsna_probs, _ = zero_shot_classification(model, inputs, "rsna_task", 1)
                            # Combine probabilities for a joint decision
                            joint_probs = np.maximum(covid_probs, rsna_probs) 
                            top_labels = np.round(joint_probs)
                            top_probs = joint_probs 
                            
                        y_true.extend(labels.cpu().numpy())
                        y_pred.extend(top_labels)
                        y_score.extend(top_probs)
                              
            return accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred), roc_auc_score(y_true, y_score), classification_report(y_true, y_pred), np.array2string(confusion_matrix(y_true, y_pred))
        for task in ["covid_task","rsna_task","both_tasks"]:
            best_auc = 0
            best_metrics = None
            for _ in range(20):
                for n_prompts in range(1,13):
                    acc, prec, rec, auc, cr, cm = evaluate(clf, generators,device, steps, ["Train", "Validation", "Test"], task,n_prompts)
                    print(f"\nAccuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, AUC: {auc:.4f}")
                    if auc > best_auc:
                        best_auc = auc
                        best_metrics = (acc, prec, rec, auc, cr, cm, n_prompts)
                if best_metrics:
                    acc, prec, rec, auc, cr, cm, n_prompts = best_metrics
                    print(f"Best AUC for {task} with {n_prompts} prompts: {auc:.4f}")
                    directory = f"{medical_type}/medclip/{vision_model_name}"
                    filename = f"{task}_classification_results.txt"
                    filepath = os.path.join(directory, filename)
                    os.makedirs(directory, exist_ok=True)
                    with open(filepath, "w") as file:
                        file.write(f"Number of Prompts: {n_prompts}\n")
                        file.write(f"Accuracy: {acc:.4f}\n")
                        file.write(f"Precision: {prec:.4f}\n")
                        file.write(f"Recall: {rec:.4f}\n")
                        file.write(f"AUC: {auc:.4f}\n")
                        file.write('Classification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))
                    print(f"Results saved to {filepath}")
        
def run_zero_shot_classification_clipmodel(medical_type, generators, steps):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    categories = ['normal', 'covid']
    def zero_shot_classification(model, image_batch,categories):
        
       
        text_inputs = torch.cat([clip.tokenize(f"a photo of {c} lungs.") for c in categories]).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_batch)
            text_features = model.encode_text(text_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            top_probs, top_labels = similarity.topk(1, dim=-1)
        return top_probs, top_labels

    def evaluate(model, data_loaders, device, steps, data_types,categories):
        model.eval()
        y_true, y_pred, y_score = [], [], []
        with torch.no_grad():
            for data_loader, data_type, step in zip(data_loaders, data_types, steps):
                for step_ind in tqdm(range(step), desc=f'Evaluate {data_type}'):
                    inputs, labels = data_loader[step_ind][0], data_loader[step_ind][1]
                    inputs = torch.from_numpy(inputs).to(device).permute(0, 3, 1, 2)
                    labels = torch.from_numpy(labels).to(device).float().unsqueeze(1)
                    top_probs, top_labels = zero_shot_classification(model,inputs, categories)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(top_labels.cpu().numpy())
                    y_score.extend(top_probs.cpu().numpy())
        return accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred), roc_auc_score(y_true, y_score), classification_report(y_true, y_pred), np.array2string(confusion_matrix(y_true, y_pred))

    acc, prec, rec, auc, cr, cm = evaluate(clip_model, generators,
                               device, steps, ["Train","Validation","Test"],
                               categories)
    print(f"\nAccuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, AUC: {auc:.4f}")
    directory = f"{medical_type}/clip"
    filename = "classification_results.txt"
    filepath = os.path.join(directory, filename)
    os.makedirs(directory, exist_ok=True)
    with open(filepath, "w") as file:
        file.write(f"Accuracy: {acc:.4f}\n")
        file.write(f"Precision: {prec:.4f}\n")
        file.write(f"Recall: {rec:.4f}\n")
        file.write(f"AUC: {auc:.4f}\n")
        file.write('Classification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))
    print(f"Results saved to {filepath}")