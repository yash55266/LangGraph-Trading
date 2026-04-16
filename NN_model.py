
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(42)

device =torch.device('cuda'if torch.cuda.is_available()else 'cpu')
print(device)

df= pd.read_csv('RELIANCE_BO_ml_data.csv')
df.shape

df.drop(['Date','High_Volume'], axis=1, inplace=True)

df.head(15)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

price_cols = ['Close', 'High', 'Low', 'Open', 'MA50', 'MA200']
volume_cols = ['Volume', 'Vol_20MA']
no_scale_cols = ['RSI', 'HL_Pct', 'Returns']

all_cols = price_cols + volume_cols + no_scale_cols

X_train_df = pd.DataFrame(X_train, columns=all_cols)
X_test_df  = pd.DataFrame(X_test,  columns=all_cols)

scaler_price = MinMaxScaler()
scaler_volume = RobustScaler()

X_train_df[price_cols]  = scaler_price.fit_transform(X_train_df[price_cols])
X_train_df[volume_cols] = scaler_volume.fit_transform(X_train_df[volume_cols])

X_test_df[price_cols]  = scaler_price.transform(X_test_df[price_cols])
X_test_df[volume_cols] = scaler_volume.transform(X_test_df[volume_cols])

X_train = X_train_df.values
X_test  = X_test_df.values


class CustomDataset(Dataset):
  def __init__(self,features,labels):
    self.features=torch.tensor(features,dtype=torch.float32)
    self.labels=torch.tensor(labels,dtype=torch.long)

  def __len__(self):
    return len(self.features)

  def __getitem__(self,index):
    return self.features[index], self.labels[index]

train_dataset=CustomDataset(X_train,y_train)

test_dataset=CustomDataset(X_test,y_test)

train_loader=DataLoader(train_dataset,batch_size=40,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=40,shuffle=False)

class MyNN(nn.Module):
  def __init__(self,num_features):
    super().__init__()

    self.model=nn.Sequential(
        nn.Linear(num_features,32),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(32,16),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(16, 1)
    )
  def forward(self,x):
    return self.model(x)

