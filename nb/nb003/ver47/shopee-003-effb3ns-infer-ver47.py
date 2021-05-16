#!/usr/bin/env python
# coding: utf-8

# ## Directory settiings

# In[1]:


# ====================================================
# Directory settings
# ====================================================
import os

OUTPUT_DIR='./'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
ROOT_DIR = '/home/yuki/shopee/input/shopee-product-matching/'
TRAIN_PATH = ROOT_DIR + 'train_images/'
TEST_PATH = ROOT_DIR + 'test_images/'


# ## CFG

# In[2]:


# ====================================================
# CFG
# ====================================================
class CFG:
    debug = False
    CHECK_SUB = False
    GET_CV = True
    num_workers = 4
    model_name_cnn = 'tf_efficientnet_b3_ns'
    model_name_bert = '/home/yuki/shopee/input/sentence-transformer-models/paraphrase-xlm-r-multilingual-v1/0_Transformer'
    size = 512
    batch_size = 8
    seed = 42
    target_size = 8811
    target_size_list = [8811, 8812, 8811, 8811, 8811]
    target_col = 'label_group'
    use_fc = False
    use_arcface = True
    scale = 30
    margin = 0.5
    fc_dim = 512
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]
    train = False
    inference = True


# In[3]:


import pandas as pd
test = pd.read_csv('/home/yuki/shopee/input/shopee-product-matching/test.csv')
if len(test)>3: 
    CFG.GET_CV = False
else: 
    print('this submission notebook will compute CV score, but commit notebook will not')


# ## Library

# In[4]:


# ====================================================
# Library
# ====================================================
import sys
sys.path.append('/home/yuki/shopee/input/timm-pytorch-image-models/pytorch-image-models-master')

import os
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import scipy as sp
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from tqdm.auto import tqdm
from functools import partial

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau, _LRScheduler

import transformers

from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, 
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose
    )
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

import gc
import matplotlib.pyplot as plt
import cudf
import cuml
import cupy
from cuml.feature_extraction.text import TfidfVectorizer
from cuml import PCA
from cuml.neighbors import NearestNeighbors

import timm

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ## Utils

# In[5]:


# ====================================================
# Utils
# ====================================================
def f1_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    f1 = 2 * intersection / (len_y_pred + len_y_true)
    return f1

def combine_predictions(row):
    x = np.concatenate([row['image_predictions'], row['text_predictions']])
    return ' '.join( np.unique(x) )

@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')

def init_logger(log_file=OUTPUT_DIR+'inference.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

#LOGGER = init_logger()

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=CFG.seed)

tokenizer = transformers.AutoTokenizer.from_pretrained(CFG.model_name_bert)


# ## Data Loading

# In[6]:


def read_dataset():
    if CFG.GET_CV:
        
        # create folds
        # trainingの時と同じようにfoldを切っています。
        folds = pd.read_csv('/home/yuki/shopee/input/shopee-product-matching/train.csv')
        if CFG.debug:
            folds = folds.sample(n=300, random_state=CFG.seed).reset_index(drop=True)  
        Fold = GroupKFold(n_splits=CFG.n_fold)
        groups = folds['label_group'].values
        for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[CFG.target_col], groups)):
            folds.loc[val_index, 'fold'] = int(n)
        folds['fold'] = folds['fold'].astype(int)
        display(folds.groupby('fold').size())
        
        tmp = folds.groupby('label_group')['posting_id'].unique().to_dict()
        folds['matches'] = folds['label_group'].map(tmp)
        folds['matches'] = folds['matches'].apply(lambda x: ' '.join(x))
        folds['file_path'] = folds['image'].apply(lambda x: TRAIN_PATH + x)
        
        if CFG.CHECK_SUB:
            folds = pd.concat([folds, folds], axis=0)
            folds.reset_index(drop=True, inplace=True)
        folds_cu = cudf.DataFrame(folds)
    else:
        folds = pd.read_csv('../input/shopee-product-matching/test.csv')
        folds['file_path'] = folds['image'].apply(lambda x: TEST_PATH + x)
        folds_cu = cudf.DataFrame(folds)
        
    return folds, folds_cu


# ## Dataset

# In[7]:


class TestDataset(Dataset):
    
    def __init__(self, df, transform=None):
        self.df = df
        self.file_paths = df['file_path'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, torch.tensor(1)


# In[8]:


class TestDataset_BERT(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.df.iloc[idx]['title']
        text = tokenizer(text, padding='max_length', truncation=True, max_length=64, return_tensors='pt')  # 'pt': pytorch
        input_ids = text['input_ids'][0]
        attention_mask = text['attention_mask'][0]
        return input_ids, attention_mask


# ## Data Loader

# In[9]:


# ====================================================
# Transforms
# ====================================================
def get_transforms(*, data):
    
    if data == 'train':
        return Compose([
            #Resize(CFG.size, CFG.size),
            RandomResizedCrop(CFG.size, CFG.size),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    elif data == 'valid':
        return Compose([
            Resize(CFG.size, CFG.size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


# ## Model

# In[10]:


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
    
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output, nn.CrossEntropyLoss()(output,label)

class CustomEfficientNet(nn.Module):
    
    def __init__(
        self,
        n_classes = CFG.target_size, 
        model_name = CFG.model_name_cnn,
        fc_dim = CFG.fc_dim,
        margin = CFG.margin,
        scale = CFG.scale,
        use_fc = True,
        pretrained = True):
        
        super(CustomEfficientNet,self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        self.pooling =  nn.AdaptiveAvgPool2d(1)
        self.use_fc = use_fc
        
        if use_fc:
            self.dropout = nn.Dropout(p=0.1)
            self.classifier = nn.Linear(in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            in_features = fc_dim

        self.final = ArcMarginProduct(
            in_features,
            n_classes,
            scale = scale,
            margin = margin,
            easy_margin = False,
            ls_eps = 0.0
        )
        
    def _init_params(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        
    def forward(self, image, label):
        features = self.extract_features(image)
        if self.training:
            logits = self.final(features, label)
            return logits
        else:
            return features
        
    def extract_features(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc and self.training:
            x = self.dropout(x)
            x = self.classifier(x)
            x = self.bn(x)
        return x


# In[11]:


class CustomBERT(nn.Module):
    def __init__(
        self,
        n_classes = CFG.target_size,
        model_name = CFG.model_name_bert,
        fc_dim = CFG.fc_dim,
        margin = CFG.margin,
        scale = CFG.scale,
        use_fc = CFG.use_fc,
        use_arcface = CFG.use_arcface,
        pretrained = True):
        
        super(CustomBERT, self).__init__()
        print(f'Building Model Backbone for {model_name} model')
        self.bert = transformers.AutoModel.from_pretrained(model_name)
        in_features = self.bert.config.hidden_size
        self.use_fc = use_fc
        self.use_arcface = use_arcface
        
        if self.use_fc:
            self.dropout = nn.Dropout(p=0.1)
            self.classifier = nn.Linear(in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            in_features = fc_dim
        
        if self.use_arcface:
            self.final = ArcMarginProduct(
            in_features,
            n_classes,
            scale = scale,
            margin = margin,
            easy_margin = False,
            ls_eps = 0.0
        )
        else:
            self.final = nn.Linear(in_features, n_classes)
    
    def _init_params(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        
    def forward(self, input_ids, attention_mask):
        features = self.extract_features(input_ids, attention_mask)
        return features
        
    def extract_features(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        features = x[0]
        features = features[:, 0, :]
        
        if self.use_fc:
            features = self.dropout(features)
            features = self.classifier(features)
            features = self.bn(features)
        return features


# ## inference functions

# In[12]:


def get_image_embeddings(folds, fold):
    
    CFG.target_size = CFG.target_size_list[fold]
    model = CustomEfficientNet(n_classes=CFG.target_size, pretrained=False).to(device)
    model_path = f'../input/shopee-002-data-local/tf_efficientnet_b3_ns_fold{fold}_best.pth'
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    
    image_dataset = TestDataset(folds, transform=get_transforms(data='valid'))
    image_loader = DataLoader(image_dataset,
                              batch_size=CFG.batch_size,
                              num_workers=CFG.num_workers,
                              pin_memory=True,
                              drop_last=False)
    embeds = []
    with torch.no_grad():
        pbar = tqdm(image_loader, total=len(image_loader))
        for img, label in pbar:
            img = img.to(device)
            label = label.to(device)
            features = model(img, label)
            image_embeddings = features.detach().cpu().numpy()
            embeds.append(image_embeddings)
            
    del model
    image_embeddings = np.concatenate(embeds)
    print(f'Our image embeddings shape is {image_embeddings.shape}')
    del embeds
    gc.collect()
    return image_embeddings


# In[13]:


def get_image_embeddings_infer(folds, fold):
    
    models = []
    for fold in CFG.trn_fold:
        CFG.target_size = CFG.target_size_list[fold]
        model = CustomEfficientNet(n_classes=CFG.target_size, pretrained=False).to(device)
        model_path = f'../input/shopee-002-data-local/tf_efficientnet_b3_ns_fold{fold}_best.pth'
        model.load_state_dict(torch.load(model_path)['model'])
        model.eval()
        models.append(model)
    
    image_dataset = TestDataset(folds, transform=get_transforms(data='valid'))
    image_loader = DataLoader(image_dataset,
                              batch_size=CFG.batch_size,
                              num_workers=CFG.num_workers,
                              pin_memory=True,
                              drop_last=False)
    embeds = []
    with torch.no_grad():
        pbar = tqdm(image_loader, total=len(image_loader))
        for img, label in pbar:
            img = img.to(device)
            label = label.to(device)
            features = []
            for model in models:
                features_ = model(img, label)
                features.append(features_.detach().cpu().numpy())
            image_embeddings = np.mean(features, axis=0)
#             image_embeddings = features
            embeds.append(image_embeddings)
            
    del model
    image_embeddings = np.concatenate(embeds)
    print(f'Our image embeddings shape is {image_embeddings.shape}')
    del embeds
    gc.collect()
    return image_embeddings


# In[14]:


def get_text_embeddings(folds, fold):
    
    CFG.target_size = CFG.target_size_list[fold]
    model = CustomBERT(n_classes=CFG.target_size, pretrained=False).to(device)
    model_path = f'../input/hopee-004-bert-training-data/paraphrase-xlm-r-multilingual-v1_fold{fold}_best.pth'
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    
    text_dataset = TestDataset_BERT(folds)
    text_loader = DataLoader(text_dataset,
                              batch_size=CFG.batch_size,
                              num_workers=CFG.num_workers,
                              pin_memory=True,
                              drop_last=False)
    embeds = []
    with torch.no_grad():
        pbar = tqdm(text_loader, total=len(text_loader))
        for input_ids, attention_mask in pbar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            features = model(input_ids, attention_mask)
            text_embeddings = features.detach().cpu().numpy()
            embeds.append(text_embeddings)
            
    del model
    text_embeddings = np.concatenate(embeds)
    print(f'Our text embeddings shape is {text_embeddings.shape}')
    del embeds
    gc.collect()
    return text_embeddings


# In[15]:


def get_text_predictions(df, df_cu, max_features=25_000):
    
    model = TfidfVectorizer(stop_words='english',
                            binary=True,
                            max_features=max_features)
    text_embeddings = model.fit_transform(df_cu['title']).toarray()
    
    print('Finding similar titles...')
    CHUNK = 1024 * 4
    CTS = len(df) // CHUNK
    if (len(df)%CHUNK) != 0:
        CTS += 1
        
    preds = []
    for j in range( CTS ):
        a = j * CHUNK
        b = (j+1) * CHUNK
        b = min(b, len(df))
        print('chunk', a, 'to', b)
        
        # COSINE SIMILARITY DISTANCE
        cts = cupy.matmul(text_embeddings, text_embeddings[a:b].T).T
        
        for k in range(b-a):
            IDX = cupy.where(cts[k,]>0.75)[0]  # 変える余地がありそう
            o = df.iloc[cupy.asnumpy(IDX)].posting_id.values
            preds.append(o)
            
    del model, text_embeddings
    gc.collect()
    return preds


# In[16]:


def get_neighbors(df, embeddings, KNN = 50, image = True, thresh_cnn=0.3):
    
    model = NearestNeighbors(n_neighbors = KNN, metric='cosine')
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)
    
    # Iterate through different thresholds to maximize cv, run this in interactive mode, then replace else clause with a solid threshold
    if CFG.GET_CV:
#         if image:
#             thresholds = list(np.arange(0.3, 0.5, 0.01))
#         else:
#             thresholds = list(np.arange(0.4, 0.6, 0.01))  # changed
#         scores = []
#         for threshold in thresholds:
#             predictions = []
#             for k in range(embeddings.shape[0]):
#                 idx = np.where(distances[k,] < threshold)[0]
#                 ids = indices[k, idx]
#                 posting_ids = ' '.join(df['posting_id'].iloc[ids].values)
#                 predictions.append(posting_ids)
#             df['pred_matches'] = predictions
#             df['f1'] = f1_score(df['matches'], df['pred_matches'])
#             score = df['f1'].mean()
#             print(f'Our f1 score for threshold {threshold} is {score}')
#             scores.append(score)
#         thresholds_scores = pd.DataFrame({'thresholds': thresholds, 'scores': scores})
#         max_score = thresholds_scores[thresholds_scores['scores'] == thresholds_scores['scores'].max()]
#         best_threshold  = max_score['thresholds'].values[0]
#         best_score = max_score['scores'].values[0]
#         print(f'Our best score is {best_score} and has a threshold {best_threshold}')
        
        # Use threshold
        predictions = []
        for k in range(embeddings.shape[0]):
            # Because we are predicting the test set that have 70K images and different label groups, confidence should be smaller
            if image:
                idx = np.where(distances[k,] < thresh_cnn)[0]
            else:
                idx = np.where(distances[k,] < 0.3)[0]
            ids = indices[k, idx]
            posting_ids = df['posting_id'].iloc[ids].values
            predictions.append(posting_ids)
            
    # Because we are predicting the test set that have 70K images and different label groups, confidence should be smaller
    else:
        predictions = []
        for k in tqdm(range(embeddings.shape[0])):
            if image:
                idx = np.where(distances[k,] < thresh_cnn)[0]
            else:
                idx = np.where(distances[k,] < 0.3)[0]
            ids = indices[k,idx]
            posting_ids = df['posting_id'].iloc[ids].values
            predictions.append(posting_ids)
        
    del model, distances, indices
    gc.collect()
    return df, predictions


# ## Calculating Predictions

# In[17]:


folds, folds_cu = read_dataset()
folds.head()


# In[ ]:


# Get neighbors for image_embeddings
if CFG.GET_CV:
    indices = []
    image_embeddings = []
    text_embeddings = []
    for fold in CFG.trn_fold:
        folds_ = folds[folds['fold'] == fold]
        folds_cu_ = folds_cu[folds['fold'] == fold]
        index = folds[folds['fold'] == fold].index
        indices.append(list(index))
        image_embeddings_ = get_image_embeddings(folds_, fold)
        image_embeddings.append(image_embeddings_)
        text_embeddings_ = get_text_embeddings(folds_, fold)
        text_embeddings.append(text_embeddings_)
    # 元のデータの順に並び替える
    indices = np.concatenate(indices)
    image_embeddings = np.concatenate(image_embeddings)
    image_embeddings = image_embeddings[indices]
    text_embeddings = np.concatenate(text_embeddings)
    text_embeddins = text_embeddins[indices]
        
    text_predictions_tfidf = get_text_predictions(folds, folds_cu, max_features=25_000)
    
    for thresh in np.arange(0.2, 0.4, 0.01):
        oof_df, image_predictions = get_neighbors(folds, image_embeddings, KNN=50 if len(folds)>3 else 3, image=True, thresh_cnn=thresh)
        oof_df, text_predictions_bert = get_neighbors(folds, text_embeddings, KNN=50 if len(folds) > 3 else 3, image=False)
        oof_df['image_predictions'] = image_predictions
        oof_df['text_predictions'] = text_predictions_tfidf
        oof_df['text_predictions_bert'] = text_predictions_bert
        oof_df['text_predictions_bert_len'] = oof_df_['text_predictions_bert'].apply(lambda x: len(x))
        oof_df['text_predictions'].mask(oof_df_['text_predictions_bert_len'] == 2, oof_df_['text_predictions_bert'], inplace=True)
        oof_df['pred_matches'] = oof_df.apply(combine_predictions, axis = 1)
        oof_df['f1'] = f1_score(oof_df['matches'], oof_df['pred_matches'])
        score = oof_df['f1'].mean()
        print(f'Our final f1 cv score for thresh {thresh} is {score}')
    oof_df.to_csv('oof_df.csv', index=False)
    oof_df[['posting_id', 'pred_matches']].to_csv('submission.csv', index = False)
        
else:
    image_embeddings = get_image_embeddings_infer(folds, fold=0)  # 後で調整する
    text_embeddings = get_text_embeddings(folds, fold=0)
    text_predictions_tfidf = get_text_predictions(folds, folds_cu, max_features=25_000) 
    df, text_predictions_bert = get_neighbors(folds, text_embeddings, KNN=50 if len(folds) > 3 else 3, image=False)
    df, image_predictions = get_neighbors(folds, image_embeddings, KNN=50 if len(folds)>3 else 3, image=True)
    df['image_predictions'] = image_predictions
    df['text_predictions_tfidf'] = text_predictions_tfidf
    df['text_predictions_bert'] = text_predictions_bert
    df['text_predictions_bert_len'] = df['text_predictions_bert'].apply(lambda x: len(x))
    df['text_predictions'] = df['text_predictions_tfidf'].mask(df['text_predictions_bert_len'] == 2, df['text_predictions_bert'])
    df['matches'] = df.apply(combine_predictions, axis = 1)
    df[['posting_id', 'matches']].to_csv('submission.csv', index = False)


# In[ ]:


pd.read_csv('submission.csv').head()

