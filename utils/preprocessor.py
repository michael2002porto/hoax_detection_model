import json
import os
import pandas as pd

from transformers import AutoTokenizer

from tqdm import tqdm

# Operasi Matrix
import torch

# Manage Data
from torch.utils.data import TensorDataset, DataLoader

import lightning as L

class Preprocessor(L.LightningDataModule):
    def __init__(self, batch_size):
        super(Preprocessor, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
        self.batch_size = batch_size
        
        self.kaggle_folder = "ferdi_indo_hoax"
        
    def load_data(self):
        dataset = pd.read_csv(f"{self.kaggle_folder}/datasets/turnbackhoax_data.csv")
        dataset = dataset[["title", "label", "narasi", "counter"]]
        
        return dataset
    
    def preprocessor(self):
        dataset = self.load_data()
        
        if not os.path.exists(f"{self.kaggle_folder}/datasets/train_set.pt") \
            and not os.path.exists(f"{self.kaggle_folder}/datasets/val_set.pt") \
            and not os.path.exists(f"{self.kaggle_folder}/datasets/test_set.pt") :
                
            x_ids, x_att, y = [], [], []
            for _, data in tqdm(dataset.iterrows(), total = dataset.shape[0], desc = "Preprocesing Hoax"):
                '''
                    1 = Real
                    0 = Fake
                '''
                
                title = data["title"]
                label = data["label"]
                narasi = data["narasi"]
                counter = data["counter"]
                
                narasi = narasi.replace("['", "").replace("']", "")
                narasi = narasi.replace("', '", ". ")
                narasi = narasi.replace(".. ",  ". ")
                
                counter = counter.replace("['", "").replace("']", "")
                counter = counter.replace("', '", ". ")
                counter = counter.replace(".. ",  ". ")
                
                narasi_tok = self.tokenizer(
                    f"{title} {narasi}",
                    max_length = 200,
                    truncation = True,
                    padding = "max_length",
                )
                
                x_ids.append(narasi_tok["input_ids"])
                x_att.append(narasi_tok["attention_mask"])
                y.append([1,0])
                
                counter_tok = self.tokenizer(
                    counter,
                    max_length = 200,
                    truncation = True,
                    padding = "max_length",
                )
                
                x_ids.append(counter_tok["input_ids"])
                x_att.append(counter_tok["attention_mask"])
                
                y.append([0,1])
            
            x_ids = torch.tensor(x_ids)
            x_att = torch.tensor(x_att)
            # y = Label (0 = Fake, 1 = Real)
            y = torch.tensor(y)
            
            # Ratio Training 80% overall data = (90% Train, 10 Validation)
            
            train_val_len = int(x_ids.shape[0] * 0.8)
            train_len = int(train_val_len * 0.9)
            val_len = train_val_len - train_len
            test_len = x_ids.shape[0] - train_val_len
            
            all_data = TensorDataset(x_ids, x_att, y)
            train_set, val_set, test_set = torch.utils.data.random_split(all_data, [train_len, val_len, test_len])
            
            torch.save(train_set, f"{self.kaggle_folder}/datasets/train_set.pt")
            torch.save(val_set, f"{self.kaggle_folder}/datasets/val_set.pt")
            torch.save(test_set, f"{self.kaggle_folder}/datasets/test_set.pt")
            
        else:
            print("Load Data")
            train_set = torch.load(f"{self.kaggle_folder}/datasets/train_set.pt")
            val_set = torch.load(f"{self.kaggle_folder}/datasets/val_set.pt")
            test_set = torch.load(f"{self.kaggle_folder}/datasets/test_set.pt")
        
        return train_set, val_set, test_set
        
    
    def setup(self, stage = None):
        # 100 data
        # 80 = Training
        # 20 = Testing
        
        train_set, val_set, test_set = self.preprocessor()
        
        if stage == "fit":
            self.train_data = train_set
            self.val_data = val_set
        elif stage == "test":
            self.test_data = test_set
            
    def train_dataloader(self):
        return DataLoader(
            self.train_data, 
            batch_size = self.batch_size, 
            shuffle = True,
            num_workers = 2,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 2,
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 2,
        )


# Deep Learning
# Data (Download, Collect) -> Preprocessing (Kalimat -> ID (token)) supaya bisa di training -> Training 
# Model = Transformers / Language Model (BERT)

if __name__ == "__main__":
    prepro = Preprocessor(batch_size = 5)
    prepro.setup(stage = "fit")
    