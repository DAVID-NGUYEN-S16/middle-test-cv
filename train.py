# %%
import numpy as np
import pandas as pd
from torch import nn
from dataset import load_data
from tqdm.notebook import tqdm
import torch
import torch.nn.functional as F
import wandb
from utils import save_model
from sklearn.metrics import f1_score
import json
from model.model import ViTModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open("/kaggle/working/middle-test-cv/config.json", 'r') as file:
    config = json.load(file)
    
wandb.login(key=config['key_wandb'])

train_loader, val_loader = load_data()

# %%
def validate(
        model,
        loader,
        loss_fn,
        data_name = "val"
):  

    model.eval()
    label_list = []
    label_pred_list = []
    val_loss = .0
    val_acc = .0
    f1_scr = .0
    with torch.no_grad():
        for batch_idx, (inputs, target) in tqdm(enumerate(loader), total = len(loader)):
            target = F.one_hot(target, config['config_model']['num_classes'])
            inputs = inputs.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float )

            output = model(inputs)
            
            
            loss = loss_fn(output, target)

            val_loss += loss.item()
            predicted = torch.max(output, 1)[1].to(device)
            label = torch.max(target, 1)[1].to(device)
            label_list.append(label.cpu())
            label_pred_list.append(predicted.cpu())

    # Tính toán độ chính xác cho toàn bộ tập dữ liệu validation
    label_list = np.concatenate([labels.numpy() for labels in label_list])
    label_pred_list = np.concatenate([preds.numpy() for preds in label_pred_list])
    val_acc = np.sum(label_list == label_pred_list) / len(label_list)
    f1_scr = f1_score(label_list, label_pred_list, average='weighted')

    metrics = {
        "data": data_name,
        "acc": round(val_acc, 3),
        "loss": round(val_loss / len(loader), 3),  # Chia cho số lượng batch để tính trung bình
        "f1": round(f1_scr, 3)
    }
    return metrics

# %%
def train_one_epoch(
        epoch,
        model,
        loader,
        optimizer,
        loss_fn,
        config=None, 
):
    train_loss = 0.
    model.train()
    label_list = []
    label_pred_list = []
    check =True
    for batch_idx, (inputs, target) in tqdm(enumerate(loader), total=len(loader)):         
        
        
        target = F.one_hot(target, config['config_model']['num_classes'])
        inputs = inputs.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float )
        
        output = model(inputs)
        
        loss = loss_fn(output, target) / config['config_model']['gradient_accumulation_steps']
        train_loss+= loss.item()
#         print(loss)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if (batch_idx+1) % config['config_model']['gradient_accumulation_steps'] ==0 :
            if check: 
                print("model have update")
                check  = False
            optimizer.step()
           
            optimizer.zero_grad()

        
        "Validation"
        
        predicted = torch.max(output, 1)[1].to(device)
        label = torch.max(target, 1)[1].to(device)
   
        label_list.append(label.cpu())
        label_pred_list.append(predicted.cpu())



    # Tính toán độ chính xác cho toàn bộ tập dữ liệu validation
    label_list = np.concatenate([labels.numpy() for labels in label_list])
    label_pred_list = np.concatenate([preds.numpy() for preds in label_pred_list])
    val_acc = np.sum(label_list == label_pred_list) / len(label_list)
    f1_scr = f1_score(label_list, label_pred_list, average='weighted')
    metrics = {
        "data": "train",
        "acc": val_acc,
        "loss": train_loss/len(loader),  # Chia cho số lượng batch để tính trung bình
        "f1": f1_scr
    }
#     print(cnt)
    return metrics


# %%
info_epoch = pd.DataFrame(columns=["epoch", "train_loss", "train_acc",  "f1-train", "val_acc","val_loss", "f1-val"])

run = wandb.init(
    # Set the project where this run will be logged
    project="Classify-Animal",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": config['config_model']['lr'],
        "epochs": config['config_model']['num_epochs'], 
        "number class":config['config_model']['num_classes']
    },
    tags = "VIT"
)

class VITmodel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.model = ViTModel()

        self.heads = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(768, output_size),
        )

    def forward(self, x):
        cls = self.model(x).pooler_output
        cls = self.heads(cls)
        return F.softmax(cls, dim=1)


# %%
def train ():
    print(f"############### Mode {config['config_model']['model_type']}  ###############")

    model = VITmodel(config["config_model"]["num_classes"])
    print(model)
    model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['config_model']['lr'], weight_decay=0.1)

    loss_fn = nn.CrossEntropyLoss().to(device=device)

    num_epochs = config['config_model']['num_epochs']

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.1)
    start_epoch = 0

    best_model = .0
    # initialize the early_stopping object
    early_stopping = 0
    for epoch in tqdm(range(start_epoch, num_epochs)):
            
        train_metrics = train_one_epoch(
            epoch=epoch,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config = config, 
        )
        lr_scheduler.step()
        

        eval_metrics = validate(
        
            model,
            val_loader,
            loss_fn,
            config=config,
            data_name="val"
        )
        
        if eval_metrics['f1'] > best_model:
            best_model = eval_metrics['f1']
            #epochs, model, optimizer, criterion, path = "./best_model.pth"
            save_model(epochs = epoch, model = model, optimizer = optimizer, criterion = loss_fn)
        else:
            early_stopping +=1
            
        if early_stopping > config['early_stopping']: break
    
        print(f"Epoch: {epoch}")
        print(f"loss train: {train_metrics['loss']}  loss val: {eval_metrics['loss']}")
        print(f"acc train: {train_metrics['acc']}  acc val: {eval_metrics['acc']} ")
        print(f"f1-train: {train_metrics['f1']}   f1-val: {eval_metrics['f1']}")
        wandb.log({
            "accuracy_train": train_metrics['acc'], 
            "accuracy_test": eval_metrics['acc'], 
            "f1_test": eval_metrics['f1'], 
            "f1_train": train_metrics['f1'],
            "loss_test": eval_metrics['loss'], 
            "loss_train": train_metrics['loss'], 
        })
        
        # "epoch", "train_loss", "train_acc",  "f1-train", "val_acc","val_loss", "f1-val", "path"]
        info_epoch.loc[epoch] = [epoch,   train_metrics['loss'], train_metrics['acc'], train_metrics['f1'], eval_metrics['acc'], eval_metrics['loss'], eval_metrics['f1']]

        

        info_epoch.to_csv(f"{config['storage_output']}/info_epoch.csv")
    

if __name__ == "__main__":

    train()  # Gọi hàm train()


