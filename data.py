import kagglehub
import pandas as pd
from torch.utils.data import Dataset
import os 
from torchvision import transforms
from PIL import Image 

# Download latest version
#path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")

#print("Path to dataset files:", path)

excel = "/Users/mayvin/.cache/kagglehub/datasets/fedesoriano/heart-failure-prediction/versions/1/heart.csv"
df = pd.read_csv(excel)



