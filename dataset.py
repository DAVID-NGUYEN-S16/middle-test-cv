from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import pandas as pd
from PIL import Image
import torch
from .train import label2id
class DatasetAnimal(Dataset):
    def __init__(self, data_path_file, train=True, global_path=None):
        super(Dataset, self).__init__()
        print("Starting dataset")
        self.data = pd.read_csv(data_path_file).sample(frac=1).reset_index()
        self.data = self.data[['path', 'label']]
        print(self.data.shape)
        if global_path == None: self.global_path = "/kaggle/input/data-food"
        else: self.global_path = global_path

        if train:
            # Define the training data augmentation pipeline
            self.transform = transforms.Compose([
                transforms.Resize(size=(224,224), interpolation = InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
            ])
        else:
            # Define the validation and testing pipeline
            self.transform = transforms.Compose([
                transforms.Resize(size=(224,224),  interpolation = InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
            ])
        print("Successfull dataset")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        label = label2id[row['label']]
        img_path = row['path']

        labels = torch.tensor(label)
        
        image = Image.open(img_path)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, labels
    
def load_data(config):
    train_dataset = DatasetAnimal(config['path_train'])
    val_dataset = DatasetAnimal(config['path_train'], train=False)
    train_loader = DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=config['batch_size'])
    return train_loader, val_loader