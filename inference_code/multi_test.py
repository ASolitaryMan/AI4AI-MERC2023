import re
import os
import sys
import time
import copy
import tqdm
import glob
import json
import math
import scipy
import shutil
import random
import pickle
import argparse
import numpy as np
import pandas as pd
import multiprocessing

import sklearn
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import multi_config as config

emos = ['neutral', 'anger', 'happy', 'sad', 'disgust',  'surprise', 'fear']
emo2idx, idx2emo = {}, {}
for ii, emo in enumerate(emos): emo2idx[emo] = ii
for ii, emo in enumerate(emos): idx2emo[ii] = emo

########################################################
############## multiprocess read features ##############
########################################################
def func_read_one(argv=None, feature_root=None, name=None):

    feature_root, name = argv
    feature_dir = glob.glob(os.path.join(feature_root, name+'.npy'))
    assert len(feature_dir) == 1
    feature_path = feature_dir[0]

    feature = []
    if feature_path.endswith('.npy'):
        single_feature = np.load(feature_path)
        single_feature = single_feature.squeeze()
        feature.append(single_feature)
    else:
        facenames = os.listdir(feature_path)
        for facename in sorted(facenames):
            facefeat = np.load(os.path.join(feature_path, facename))
            feature.append(facefeat)

    single_feature = np.array(feature).squeeze()
    if len(single_feature) == 0:
        print ('feature has errors!!')
    elif len(single_feature.shape) == 2:
        single_feature = np.mean(single_feature, axis=0)
    return single_feature
    
def read_data_multiprocess(label_path, feature_root, task='emo', data_type='train', debug=False):

    ## gain (names, labels)
    names, labels = [], []
    assert task in  ['emo', 'aro', 'val', 'whole']
    assert data_type in ['train', 'test3']
    if data_type == 'train': corpus = np.load(label_path, allow_pickle=True)['train_corpus'].tolist()
    if data_type == 'test3': corpus = np.load(label_path, allow_pickle=True)['test3_corpus'].tolist()
    for name in corpus:
        names.append(name)
        if task in ['aro', 'val']:
            labels.append(corpus[name][task])
        if task == 'emo':
            labels.append(emo2idx[corpus[name]['emo']])
        if task == 'whole':
            corpus[name]['emo'] = emo2idx[corpus[name]['emo']]
            labels.append(corpus[name])

    ## ============= for debug =============
    if debug: 
        names = names[:100]
        labels = labels[:100]
    ## =====================================

    ## names => features
    params = []
    for ii, name in tqdm.tqdm(enumerate(names)):
        params.append((feature_root, name))

    features = []
    with multiprocessing.Pool(processes=8) as pool:
        features = list(tqdm.tqdm(pool.imap(func_read_one, params), total=len(params)))
    feature_dim = np.array(features).shape[-1]

    ## save (names, features)
    print (f'Input feature {feature_root} ===> dim is {feature_dim}')
    assert len(names) == len(features), f'Error: len(names) != len(features)'
    name2feats, name2labels = {}, {}
    for ii in range(len(names)):
        name2feats[names[ii]]  = features[ii]
        name2labels[names[ii]] = labels[ii]
    return name2feats, name2labels, feature_dim


########################################################
##################### data loader ######################
########################################################
class MERDataset(Dataset):

    def __init__(self, label_path, audio_root, text_root, video_root, data_type, debug=False):
        assert data_type in ['train', 'test3']
        self.name2audio, self.name2labels, self.adim = read_data_multiprocess(label_path, audio_root, task='whole', data_type=data_type, debug=debug)
        self.name2text,  self.name2labels, self.tdim = read_data_multiprocess(label_path, text_root,  task='whole', data_type=data_type, debug=debug)
        self.name2video, self.name2labels, self.vdim = read_data_multiprocess(label_path, video_root, task='whole', data_type=data_type, debug=debug)
        self.names = [name for name in self.name2audio if 1==1]

    def __getitem__(self, index):
        name = self.names[index]
        return torch.FloatTensor(self.name2audio[name]),\
               torch.FloatTensor(self.name2text[name]),\
               torch.FloatTensor(self.name2video[name]),\
               self.name2labels[name]['emo'],\
               name

    def __len__(self):
        return len(self.names)

    def get_featDim(self):
        print (f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}')
        return self.adim, self.tdim, self.vdim


def get_loaders(args, config):
    test_loaders = []
    for test_set in args.test_sets:
        test_dataset = MERDataset(label_path = config.PATH_TO_LABEL[args.test_dataset],
                                  audio_root = os.path.join(config.PATH_TO_FEATURES[args.test_dataset], args.audio_feature),
                                  text_root  = os.path.join(config.PATH_TO_FEATURES[args.test_dataset], args.text_feature),
                                  video_root = os.path.join(config.PATH_TO_FEATURES[args.test_dataset], args.video_feature),
                                  data_type  = test_set,
                                  debug      = args.debug)
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=False,
                                 pin_memory=False)
        test_loaders.append(test_loader)

    # return loaders
    adim, tdim, vdim = test_dataset.get_featDim()
    return test_loaders, adim, tdim, vdim

########################################################
##################### build model ######################
########################################################
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim1, output_dim2=1, layers='256,128', dropout=0.3):
        super(MLP, self).__init__()

        self.all_layers = []
        layers = list(map(lambda x: int(x), layers.split(',')))
        for i in range(0, len(layers)):
            self.all_layers.append(nn.Linear(input_dim, layers[i]))
            self.all_layers.append(nn.ReLU())
            self.all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        self.module = nn.Sequential(*self.all_layers)
        self.fc_out_1 = nn.Linear(layers[-1], output_dim1)
        self.fc_out_2 = nn.Linear(layers[-1], output_dim2)
        
    def forward(self, inputs):
        features = self.module(inputs)
        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        return features, emos_out, vals_out


class Attention(nn.Module):
    def __init__(self, audio_dim, text_dim, video_dim, output_dim1, output_dim2=1, layers='256,128', dropout=0.3):
        super(Attention, self).__init__()

        self.audio_mlp = self.MLP(audio_dim, layers, dropout)
        self.text_mlp  = self.MLP(text_dim,  layers, dropout)
        self.video_mlp = self.MLP(video_dim, layers, dropout)

        layers_list = list(map(lambda x: int(x), layers.split(',')))
        hiddendim = layers_list[-1] * 3
        self.attention_mlp = self.MLP(hiddendim, layers, dropout)

        self.fc_att   = nn.Linear(layers_list[-1], 3)
        self.fc_out_1 = nn.Linear(layers_list[-1], output_dim1)
        self.fc_out_2 = nn.Linear(layers_list[-1], output_dim2)
    
    def MLP(self, input_dim, layers, dropout):
        all_layers = []
        layers = list(map(lambda x: int(x), layers.split(',')))
        for i in range(0, len(layers)):
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.ReLU())
            all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        module = nn.Sequential(*all_layers)
        return module

    def forward(self, audio_feat, text_feat, video_feat):
        audio_hidden = self.audio_mlp(audio_feat) # [32, 128]
        text_hidden  = self.text_mlp(text_feat)   # [32, 128]
        video_hidden = self.video_mlp(video_feat) # [32, 128]

        multi_hidden1 = torch.cat([audio_hidden, text_hidden, video_hidden], dim=1) # [32, 384]
        attention = self.attention_mlp(multi_hidden1)
        attention = self.fc_att(attention)
        attention = torch.unsqueeze(attention, 2) # [32, 3, 1]

        multi_hidden2 = torch.stack([audio_hidden, text_hidden, video_hidden], dim=2) # [32, 128, 3]
        fused_feat = torch.matmul(multi_hidden2, attention)
        fused_feat = fused_feat.squeeze() # [32, 128]
        emos_out  = self.fc_out_1(fused_feat)
        vals_out  = self.fc_out_2(fused_feat)
        return fused_feat, emos_out, vals_out


class CELoss(nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.NLLLoss(reduction='sum')

    def forward(self, pred, target):
        pred = F.log_softmax(pred, 1)
        target = target.squeeze().long()
        loss = self.loss(pred, target) / len(pred)
        return loss

class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target):
        pred = pred.view(-1,1)
        target = target.view(-1,1)
        loss = self.loss(pred, target) / len(pred)
        return loss


########################################################
########### main training/testing function #############
########################################################
def train_or_eval_model(args, model, reg_loss, cls_loss, dataloader, optimizer=None, train=False):
    
    vidnames = []
    emo_probs, emo_labels = [], []
    embeddings = []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        ## analyze dataloader
        audio_feat, text_feat, visual_feat = data[0], data[1], data[2]
        emos = data[3].float()
        vidnames += data[-1]
        multi_feat = torch.cat([audio_feat, text_feat, visual_feat], dim=1)
        
        ## add cuda
        emos = emos.cuda()
        audio_feat  = audio_feat.cuda()
        text_feat   = text_feat.cuda()
        visual_feat = visual_feat.cuda()
        multi_feat  = multi_feat.cuda()

        ## feed-forward process
        if args.model_type == 'mlp':
            features, emos_out, vals_out = model(multi_feat)
        elif args.model_type == 'attention':
            features, emos_out, vals_out = model(audio_feat, text_feat, visual_feat)
            if len(emos_out.size()) == 1:
                emos_out = emos_out.view(-1, 7)
            if len(features.size()) == 1:
                features = features.view(-1, 128)
        emo_probs.append(emos_out.data.cpu().numpy())
        emo_labels.append(emos.data.cpu().numpy())
        embeddings.append(features.data.cpu().numpy())

        ## optimize params
        if train:
            loss = cls_loss(emos_out, emos)
            loss.backward()
            optimizer.step()

    ## evaluate on discrete labels
    emo_probs  = np.concatenate(emo_probs)
    embeddings = np.concatenate(embeddings)
    emo_labels = np.concatenate(emo_labels)
    emo_preds = np.argmax(emo_probs, 1)
    emo_accuracy = accuracy_score(emo_labels, emo_preds)
    emo_fscore = f1_score(emo_labels, emo_preds, average='weighted')

    save_results = {}
    # item1: statistic results
    save_results['emo_fscore'] = emo_fscore
    save_results['emo_accuracy'] = emo_accuracy
    # item2: sample-level results
    save_results['emo_probs'] = emo_probs
    save_results['emo_labels'] = emo_labels
    save_results['names'] = vidnames
    # item3: latent embeddings
    if args.savewhole: save_results['embeddings'] = embeddings
    return save_results


########################################################
############# metric and save results ##################
########################################################
def overall_metric(emo_fscore):
    final_score = emo_fscore
    return final_score

def average_folder_results(folder_save, testname):
    name2preds = {}
    num_folder = len(folder_save)
    for ii in range(num_folder):
        names    = folder_save[ii][f'{testname}_names']
        emoprobs = folder_save[ii][f'{testname}_emoprobs']
        for jj in range(len(names)):
            name = names[jj]
            emoprob = emoprobs[jj]
            if name not in name2preds: name2preds[name] = []
            name2preds[name].append({'emo': emoprob})

    ## gain average results
    name2avgpreds = {}
    for name in name2preds:
        preds = np.array(name2preds[name])
        emoprobs = [pred['emo'] for pred in preds if 1==1]

        avg_emoprob = np.mean(emoprobs, axis=0)
        avg_emopred = np.argmax(avg_emoprob)
        name2avgpreds[name] = {'emo': avg_emopred, 'emoprob': avg_emoprob}
    return name2avgpreds

def gain_name2feat(folder_save, testname):
    name2feat = {}
    assert len(folder_save) >= 1
    names      = folder_save[0][f'{testname}_names']
    embeddings = folder_save[0][f'{testname}_embeddings']
    for jj in range(len(names)):
        name = names[jj]
        embedding = embeddings[jj]
        name2feat[name] = embedding
    return name2feat

def write_to_csv_pred(name2preds, save_path):
    names, emos, vals = [], [], []
    for name in name2preds:
        names.append(name)
        emos.append(idx2emo[name2preds[name]['emo']])
    
    names.append("B_xiaohuanxi_20_18")
    emos.append(idx2emo[name2preds["B_xiaohuanxi_20_19"]['emo']])

    columns = ['utterance_id', 'emotion']
    data = np.column_stack([names, emos])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(save_path, index=False)


## only fscore for test3
def report_results_on_test3(test_label, test_pred):
    # read target file (few for test3)
    name2label = {}
    df_label = pd.read_csv(test_label)
    for _, row in df_label.iterrows():
        name = row['utterance_id']
        emo  = row['emotion']
        name2label[name] = {'emo': emo2idx[emo]}
    print (f'labeled samples: {len(name2label)}')

    # read prediction file (more for test3)
    name2pred = {}
    df_label = pd.read_csv(test_pred)
    for _, row in df_label.iterrows():
        name = row['utterance_id']
        emo  = row['emotion']
        name2pred[name] = {'emo': emo2idx[emo]}
    print (f'predict samples: {len(name2pred)}')
    assert len(name2pred) >= len(name2label)

    emo_labels, emo_preds = [], []
    for name in name2label: # on few for test3
        emo_labels.append(name2label[name]['emo'])
        emo_preds.append(name2pred[name]['emo'])

    # analyze results
    emo_fscore = f1_score(emo_labels, emo_preds, average='weighted')
    print (f'emo results (weighted f1 score): {emo_fscore:.4f}')
    return emo_fscore, -100, -100


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Params for input
    parser.add_argument('--dataset', type=str, default=None, help='dataset')
    parser.add_argument('--train_dataset', type=str, default=None, help='dataset') # for cross-dataset evaluation
    parser.add_argument('--test_dataset',  type=str, default=None, help='dataset') # for cross-dataset evaluation
    parser.add_argument('--audio_feature', type=str, default=None, help='audio feature name')
    parser.add_argument('--text_feature', type=str, default=None, help='text feature name')
    parser.add_argument('--video_feature', type=str, default=None, help='video feature name')
    parser.add_argument('--debug', action='store_true', default=False, help='whether use debug to limit samples')
    parser.add_argument('--test_sets', type=str, default='test3', help='process on which test sets, [test1, test2, test3]')
    parser.add_argument('--save_root', type=str, default='./saved', help='save prediction results and models')
    parser.add_argument('--savewhole', action='store_true', default=False, help='whether save latent embeddings')

    ## Params for model
    parser.add_argument('--layers', type=str, default='256,128', help='hidden size in model training')
    parser.add_argument('--n_classes', type=int, default=-1, help='number of classes [defined by args.label_path]')
    parser.add_argument('--num_folder', type=int, default=-1, help='folders for cross-validation [defined by args.dataset]')
    parser.add_argument('--model_type', type=str, default='mlp', help='model type for training [mlp or attention]')

    ## Params for training
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, metavar='nw', help='number of workers')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--seed', type=int, default=100, help='make split manner is same with same seed')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
    parser.add_argument('--model_path', default="trimodel", type=str, help='GPU id to use')
    args = parser.parse_args()

    args.n_classes = 7
    args.num_folder = 5
    args.test_sets = args.test_sets.split(',')
    max_eval_metric = -100

    if args.dataset is not None:
        args.train_dataset = args.dataset
        args.test_dataset  = args.dataset
    assert args.train_dataset is not None
    assert args.test_dataset  is not None

    whole_features = [args.audio_feature, args.text_feature, args.video_feature]
    if len(set(whole_features)) == 1:
        args.save_root = f'{args.save_root}-unimodal'
    elif len(set(whole_features)) == 2:
        args.save_root = f'{args.save_root}-bimodal'
    elif len(set(whole_features)) == 3:
        args.save_root = f'{args.save_root}-trimodal'

    torch.cuda.set_device(args.gpu)
    print(args)

    
    print (f'====== Reading Data =======')
    test_loaders, adim, tdim, vdim = get_loaders(args, config)      

    print (f'====== Evaluation =======')
    print (f'Step1: build model (each folder has its own model)')
    if args.model_type == 'mlp':
        model = MLP(input_dim=adim + tdim + vdim,
                        output_dim1=args.n_classes,
                        output_dim2=1,
                        layers=args.layers)
    elif args.model_type == 'attention':
        model = Attention(audio_dim=adim,
                              text_dim=tdim,
                              video_dim=vdim,
                              output_dim1=args.n_classes,
                              output_dim2=1,
                              layers=args.layers)
    reg_loss = MSELoss()
    cls_loss = CELoss()
    model.cuda()
    reg_loss.cuda()
    cls_loss.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    print (f'Step2: resotring model')
    test_save = []
    for i in range(args.num_folder):
        store_values = {}
        name_time  = time.time()
        model_path = args.model_path + str(i) + "_model.pt"
        model.load_state_dict(torch.load(model_path))
        
        for jj, test_loader in enumerate(test_loaders):
            test_set = args.test_sets[jj]
            test_results = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, optimizer=None, train=False)
            store_values[f'{test_set}_emoprobs']   = test_results['emo_probs']
            store_values[f'{test_set}_names']      = test_results['names']
            if args.savewhole: store_values[f'{test_set}_embeddings'] = test_results['embeddings']
        test_save.append(store_values)


    print (f'====== Gain predition on test data =======')
    save_modelroot = os.path.join(args.save_root, 'model')
    save_predroot  = os.path.join(args.save_root, 'prediction')
    if not os.path.exists(save_predroot): os.makedirs(save_predroot)
    if not os.path.exists(save_modelroot): os.makedirs(save_modelroot)
    feature_name = f'{args.audio_feature}+{args.text_feature}+{args.video_feature}'

    for setname in args.test_sets:
        pred_path  = f'{save_predroot}/{setname}-pred-{name_time}.csv'
        label_path = f'./dataset-release/{setname}-label.csv'

        name2preds = average_folder_results(test_save, setname)
        if args.savewhole: name2feats = gain_name2feat(test_save, setname)
        write_to_csv_pred(name2preds, pred_path)

        res_name = 'nores'
        if os.path.exists(label_path):
            if setname in ['test3']:          
                emo_fscore, final_metric = report_results_on_test3(label_path, pred_path)
                res_name = f'f1:{emo_fscore:.4f}_metric:{final_metric:.4f}'

        save_path = f'{save_modelroot}/{setname}_features:{feature_name}_{res_name}_{name_time}.npz'
        print (f'save results in {save_path}')
        print (f'save pred in {pred_path}')
        if args.savewhole:
            np.savez_compressed(save_path,
                            name2preds=name2preds,
                            name2feats=name2feats,
                            args=np.array(args, dtype=object))
        else:
            np.savez_compressed(save_path,
                                name2preds=name2preds,
                                args=np.array(args, dtype=object))