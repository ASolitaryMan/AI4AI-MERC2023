import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch
import torch.nn.functional as F
import torch.optim as optim

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
    Wav2Vec2Model,
    HubertPreTrainedModel,
    AutoConfig
)
import random
from sklearn.utils.class_weight import compute_class_weight
import soundfile as sf
import torch.nn as nn
from accelerate import Accelerator
import time
import argparse
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import mean_squared_error
from accelerate import DistributedDataParallelKwargs
from tqdm import tqdm
import soundfile as sf
from fine_tune_macbert import TextForSequenceClassification
from transformers import BertTokenizer


emos = ['Neutral', 'Anger', 'Happy', 'Sad', 'Disgust',  'Surprise', 'Fear']
emo2idx, idx2emo = {}, {}
for ii, emo in enumerate(emos): emo2idx[emo] = ii
for ii, emo in enumerate(emos): idx2emo[ii] = emo

class HubertClassificationHead(nn.Module):
    """Head for hubert classification task."""

    def __init__(self, dropout=0.3):
        super().__init__()
        # self.dense1 = nn.Sequential(
        #     nn.Linear(1024, 256),
        #     nn.ReLU(),
        #     nn.Dropout(dropout)
        # )

        # self.dense2 = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Dropout(dropout)
        # )

        self.fc_out_1 = nn.Linear(1024, 7)

    def forward(self, features):
        # x = self.dense1(features)
        # feat = self.dense2(x)
        emos_out  = self.fc_out_1(features)
        return emos_out
    
class HubertForClassification(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.hubert = HubertModel(config)
        self.hubert.encoder.gradient_checkpointing = False
        # self.hubert.config.mask_feature_prob = mask_feature_prob
        # self.hubert.config.mask_time_prob = mask_time_prob
        # self.hubert.config.mask_time_length = mask_time_length
        # self.hubert.config.mask_feature_length = mask_feature_length

        self.dropout = nn.Dropout(self.hubert.config.final_dropout)

        self.pooling_mode = "mean"

        self.classifier = HubertClassificationHead(dropout=0.3)
    
    def freeze_feature_extractor(self):
        self.hubert.feature_extractor._freeze_parameters()
    
    def freeze_hubert(self):
        for param in self.hubert.parameters():
            param.requires_grad = False

    def merged_strategy(self, hidden_states, mask, mode="mean"):
        if mode == "mean":
            outputs = hidden_states.sum(dim=1) / mask.sum(dim=1).view(-1, 1)
        elif mode == "atten":
            outputs = torch.sum(hidden_states, dim=1)
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'atten']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        padding_mask = self.hubert._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)

        hidden_states[~padding_mask] = 0.0
        hidden_states = self.merged_strategy(hidden_states, padding_mask, mode=self.pooling_mode)
        emos_out = self.classifier(hidden_states)

        return emos_out

def inferenc_wav(file_src, model, feature_extractor):
    utt_idx = []
    utt_pred = []
    model.eval()

    for item in file_src:
        tmp = item.strip().split("\n")[0].split()
        
        samples, sr = sf.read(tmp[1])
        inputs = feature_extractor(samples, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", return_attention=True)
        input_values = inputs["input_values"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            emos_out = model(input_values=input_values, attention_mask=attention_mask)
        emos_out = emos_out.data.cpu().numpy()
        emo_preds = np.argmax(emos_out, 1)
        pred = idx2emo[emo_preds[0]]

        name_list = tmp[0].split("_")[1:]
        for i in range(len(name_list[2:])):
            name = "_".join(name_list[:2] + [name_list[2:][i]]) 
            print(name, pred)
        #A_luohunshidai_1_9_10_11_12_13
        #name = "_".join(tmp[0].split("_")[1:])
        # print(name, pred)


def inferenc_text(file_src, model, tokenizer):
    utt_idx = []
    utt_pred = []
    model.eval()
    for item in file_src:
        tmp = item.strip().split("\n")[0].split()
        
        inputs = tokenizer(tmp[-1], return_tensors='pt')
        input_values = inputs["input_ids"].to(device)

        with torch.no_grad():
            emos_out = model(input_ids=input_values)
        emos_out = emos_out.data.cpu().numpy()
        emo_preds = np.argmax(emos_out, 1)
        pred = idx2emo[emo_preds[0]]

        name_list = tmp[0].split("_")[1:]
        for i in range(len(name_list[2:])):
            name = "_".join(name_list[:2] + [name_list[2:][i]]) 
            print(name, pred)
        #A_luohunshidai_1_9_10_11_12_13
        #name = "_".join(tmp[0].split("_")[1:])
        # print(name, pred)


def read_text(text_path):
    f = open(text_path, 'r')
    lines = f.readlines()
    f.close()

    return lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run.')
    # choose one from supported models. Note: some transformers listed above are not available due to various errors!
    parser.add_argument('--gpu', type=str, default='1', help='gpu id')
    parser.add_argument('--overwrite', action='store_true', default=True, help='whether overwrite existed feature folder.')
    parser.add_argument('--trans_path', type=str, default='train.csv', help='name of feature level, FRAME or UTTERANCE')
    parser.add_argument('--feat_path', type=str, default='./features/', help='name of feature level, FRAME or UTTERANCE')
    parser.add_argument('--model_path', type=str, default='./saved_text', help='name of feature level, FRAME or UTTERANCE')
    parser.add_argument('--modal', type=str, default='audio', help='name of feature level, FRAME or UTTERANCE')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')

    if args.model_path.find('wav') != -1:
        #语音推理
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_path)
        model = HubertForClassification.from_pretrained(args.model_path)
    elif args.model_path.find("text") != -1:
        tokenizer = BertTokenizer.from_pretrained("./tools/transformers/chinese-macbert-large")
        model = TextForSequenceClassification.from_pretrained(args.model_path)
    else:
        #多模态推理
        pass
    
    model.to(device)
    if os.path.isfile(args.feat_path):
        lines = read_text(args.feat_path)
        if args.modal == 'audio':
            inferenc_wav(lines, model, feature_extractor)
        if args.modal == "text":
            inferenc_text(lines, model, tokenizer)
    else:
        pass







    






