import torch
from torch import optim
import model_evaluation.clip_utils as clip_utils
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
from medclip import MedCLIPModel, MedCLIPVisionModelViT

status = os.system('ldconfig /usr/lib64-nvidia')
if status == 0:
    print("Command executed successfully")
else:
    print("Error executing the command")

torch._dynamo.config.suppress_errors = True

def convert_models_to_fp32(model):
    for p in model.parameters():
        if p.grad is not None:
            p.data = p.data.float()
            p.grad.data = p.grad.data.float()

def zero_shot_classification(model, image_batch, categories, device):
    text_inputs = torch.cat([clip_utils.tokenize(f"a photo of {c} lungs.") for c in categories]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_batch)
        text_features = model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_probs, top_labels = similarity.topk(1, dim=-1)
    return top_probs, top_labels

def evaluate(model, data_loader, device, categories):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            top_probs, top_labels = zero_shot_classification(model, inputs, categories, device)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(top_labels.squeeze().cpu().numpy())
            y_score.extend(top_probs.squeeze().cpu().numpy())
    return accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred), roc_auc_score(y_true, y_score), classification_report(y_true, y_pred), confusion_matrix(y_true, y_pred)

def train_clip(medical_type, batch_size, train_generator, validation_generator, train_length, validation_length, EPOCH=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip_utils.load("ViT-B/32", device=device, jit=False)
    
    if device == "cpu":
        model.float()
    else:
        clip_utils.model.convert_weights(model)
    
    optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    
    for epoch in range(EPOCH):
        model.train()
        for images, texts in tqdm(train_generator, desc=f'Epoch {epoch+1}/{EPOCH}'):
            optimizer.zero_grad()
            images, texts = images.to(device), texts.to(device)
            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(images.size(0), dtype=torch.long, device=device)
            total_loss = (torch.nn.CrossEntropyLoss()(logits_per_image, ground_truth) + torch.nn.CrossEntropyLoss()(logits_per_text, ground_truth)) / 2
            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else : 
                convert_models_to_fp32(model)
                optimizer.step()
                clip_utils.model.convert_weights(model)

    categories = ["Category1", "Category2"]  # Placeholder for actual categories
    acc, prec, rec, auc, cr, cm = evaluate(model, validation_generator, device, categories)
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
        





