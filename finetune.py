import torch
from torch import optim
import clip
import os
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel
from medclip import PromptClassifier

status = os.system('ldconfig /usr/lib64-nvidia')
if status == 0:
    print("Command executed successfully")
else:
    print("Error executing the command")
torch._dynamo.config.suppress_errors = True
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 
def train_clip(medical_type, generators, lengths):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for vision_model_name,vision_model in zip(["MedCLIPVisionModelViT"],[MedCLIPVisionModelViT]):
        model = MedCLIPModel(vision_cls=vision_model)
        model.from_pretrained()
        model.to(device)
        clf = PromptClassifier(model, ensemble=True)
        clf.to(device)
    
    def train():
        
    def evaluation():

def train_medclip(medical_type, batch_size, train_generator, validation_generator, train_length, validation_length):
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
    
    def train():
        if device == "cpu":
  model.float()
else :
  clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

loss_img = torch.CrossEntropyLoss()
loss_txt = torch.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

# add your own code to track the training progress.
for epoch in range(EPOCH):
  for batch in train_dataloader :
      optimizer.zero_grad()

      images,texts = batch 
    
      images= images.to(device)
      texts = texts.to(device)
    
      logits_per_image, logits_per_text = model(images, texts)

      ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

      total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
      total_loss.backward()
      if device == "cpu":
         optimizer.step()
      else : 
        convert_models_to_fp32(model)
        optimizer.step()
        clip.model.convert_weights(model)
        
    def evaluation():
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
                               device, steps, ["Test"],
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
        





