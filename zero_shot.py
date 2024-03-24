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

def run_zero_shot_classification_medclipmodel(medical_type, batch_size, train_generator, validation_generator, test_generator, 
                                 train_length, validation_length, test_length):

    steps_per_epoch_training = train_length // batch_size
    steps_per_epoch_validation = validation_length // batch_size
    steps_per_epoch_test = test_length // batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    model.from_pretrained()
    model.to(device)
    clf = PromptClassifier(model, ensemble=True)
    clf.to(device)

    def zero_shot_classification(model, image_batch):
        input_dictionary = {'pixel_values': image_batch}
        cls_prompts = process_class_prompts(generate_rsna_class_prompts(n=100))
        input_dictionary['prompt_inputs'] = cls_prompts
        output = model(**input_dictionary)['logits'].cpu().numpy()
        top_probs = output.reshape(1, -1)[0]
        top_labels = np.round(top_probs)
        return top_probs, top_labels

    def evaluate(model, data_loaders, device, steps, data_types):
        model.eval()
        y_true, y_pred, y_score = [], [], []
        with torch.no_grad():
            for data_loader, data_type, step in zip(data_loaders, data_types, steps):
                for step_ind in tqdm(range(step), desc=f'Evaluate {data_type}'):
                    inputs, labels = data_loader[step_ind][0], data_loader[step_ind][1]
                    inputs = torch.from_numpy(inputs).to(device)
                    labels = torch.from_numpy(labels).to(device).float().unsqueeze(1)
                    top_probs, top_labels = zero_shot_classification(model,inputs)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(top_labels)
                    y_score.extend(top_probs)
        return accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred), roc_auc_score(y_true, y_score), classification_report(y_true, y_pred), np.array2string(confusion_matrix(y_true, y_pred))

    acc, prec, rec, auc, cr, cm = evaluate(clf, [train_generator, validation_generator, test_generator],
                                   device, [steps_per_epoch_training, steps_per_epoch_validation, steps_per_epoch_test],
                                   ["Train", "Validation", "Test"])
    print(f"\nAccuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, AUC: {auc:.4f}")
    with open(f"{medical_type}_medclip_classification_results.txt", "w") as file:
        file.write(f"Accuracy: {acc:.4f}\n")
        file.write(f"Precision: {prec:.4f}\n")
        file.write(f"Recall: {rec:.4f}\n")
        file.write(f"AUC: {auc:.4f}\n")
        file.write('Classification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))
        
def run_zero_shot_classification_clipmodel(medical_type, batch_size, train_generator, validation_generator, test_generator, 
                                 train_length, validation_length, test_length):

    steps_per_epoch_training = train_length // batch_size
    steps_per_epoch_validation = validation_length // batch_size
    steps_per_epoch_test = test_length // batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    categories = ['normal', 'pneumonia']
    def zero_shot_classification(model, image_batch,categories):
        text_inputs = torch.cat([clip.tokenize(f"An X-Ray image of a {c} lung") for c in categories]).to(device)
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

    acc, prec, rec, auc, cr, cm = evaluate(clip_model, [train_generator,validation_generator,test_generator],
                               device,[steps_per_epoch_training,steps_per_epoch_validation,steps_per_epoch_test], ["Train","Validation","Test"],
                               categories)
    print(f"\nAccuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, AUC: {auc:.4f}")
    with open(f"{medical_type}_clip_classification_results.txt", "w") as file:
        file.write(f"Accuracy: {acc:.4f}\n")
        file.write(f"Precision: {prec:.4f}\n")
        file.write(f"Recall: {rec:.4f}\n")
        file.write(f"AUC: {auc:.4f}\n")
        file.write('Classification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))