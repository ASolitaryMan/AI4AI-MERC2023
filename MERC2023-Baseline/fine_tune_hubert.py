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
import math
from torch.utils.data.sampler import Sampler
from copy import deepcopy
from torch.utils.data import WeightedRandomSampler
from ignite.distributed import DistributedProxySampler
from torch.utils.data import DataLoader
from exhaustive_weighted_random_sampler import ExhaustiveWeightedRandomSampler
from transformers import PerceiverForSequenceClassification


emos = ['Neutral', 'Anger', 'Happy', 'Sad', 'Disgust',  'Surprise', 'Fear']
emo2idx, idx2emo = {}, {}
for ii, emo in enumerate(emos): emo2idx[emo] = ii
for ii, emo in enumerate(emos): idx2emo[ii] = emo

def read_text(text_path):
    f = open(text_path, "r")
    lines = f.readlines()
    f.close()
    return lines

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`)
            The feature_extractor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List) -> Dict[str, torch.Tensor]:
        input_features, emo_label = [], []
        sample_rate = self.feature_extractor.sampling_rate

        for feature in features:
            input_features.append({"input_values": feature[0]})
            emo_label.append(feature[1])

        
        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length * sample_rate,
            pad_to_multiple_of=self.pad_to_multiple_of,
            truncation=True,
            return_tensors="pt",
        )

        d_type = torch.long if isinstance(emo_label[0], int) else torch.float32
        batch["emo_labels"] = torch.tensor(emo_label, dtype=d_type)

        return batch


class DistributedWeightedSampler(Sampler):
    """
    A class for distributed data sampling with weights.

    .. note::

        For this to work correctly, global seed must be set to be the same across
        all devices.

    :param weights: A list of weights to sample with.
    :type weights: list
    :param num_samples: Number of samples in the dataset.
    :type num_samples: int
    :param replacement: Do we sample with or without replacement.
    :type replacement: bool
    :param num_replicas: Number of processes running training.
    :type num_replicas: int
    :param rank: Current device number.
    :type rank: int
    """

    def __init__(
        self,
        weights: list,
        num_samples: int = None,
        replacement: bool = True,
        num_replicas: int = None,
    ):
        if num_replicas is None:
            num_replicas = torch.cuda.device_count()

        self.num_replicas = num_replicas
        self.num_samples_per_replica = int(
            math.ceil(len(weights) * 1.0 / self.num_replicas)
        )
        self.total_num_samples = self.num_samples_per_replica * self.num_replicas
        self.weights = weights
        self.replacement = replacement

    def __iter__(self):
        """
        Produces mini sample list for current rank.

        :returns: A generator of samples.
        :rtype: Generator
        """
        rank = os.environ["LOCAL_RANK"]

        rank = int(rank)

        if rank >= self.num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in "
                "the interval [0, {}]".format(rank, self.num_replicas - 1)
            )

        # weights = self.weights.copy()
        weights = deepcopy(self.weights)
        # add extra samples to make it evenly divisible
        weights += weights[: (self.total_num_samples) - len(weights)]
        if not len(weights) == self.total_num_samples:
            raise RuntimeError(
                "There is a distributed sampler error. Num weights: {}, total size: {}".format(
                    len(weights), self.total_size
                )
            )

        # subsample for this rank
        weights = weights[rank : self.total_num_samples : self.num_replicas]
        weights_used = [0] * self.total_num_samples
        weights_used[rank : self.total_num_samples : self.num_replicas] = weights

        return iter(
            torch.multinomial(
                input=torch.as_tensor(weights_used, dtype=torch.double),
                num_samples=self.num_samples_per_replica,
                replacement=self.replacement,
            ).tolist()
        )

    def __len__(self):
        return self.num_samples_per_replica

class MERDataset(Dataset):

    def __init__(self, src_path):
        all_lines = read_text(src_path)
        self.label = []
        self.wav_list = []
        for line in all_lines:
            tmp = line.strip().split("\n")[0].split()
            self.wav_list.append(tmp[1])
            self.label.append(emo2idx[tmp[-1]])
        
    def __getitem__(self, index):

        wave, sr = sf.read(self.wav_list[index])
        assert sr == 16000
        lab = self.label[index]
    
        return torch.FloatTensor(wave), lab

    def __len__(self):
        return len(self.label)
    
    def class_weight_v(self):
        labels = np.array(self.label)
        class_weight = torch.tensor([1/x for x in np.bincount(labels)], dtype=torch.float32)
        return class_weight
    
    def class_weight_q(self):
        class_weight = self.class_weight_v()
        return class_weight / class_weight.sum()
    
    def class_weight_k(self):
        labels = np.array(self.label)
        class_sample_count = np.unique(labels, return_counts=True)[1]
        weight = 1. / class_sample_count
        weight = weight.tolist()
        samples_weight = torch.tensor([weight[t] for t in labels], dtype=torch.float32)
        """
        class_sample_count = np.unique(labels, return_counts=True)[1]
        class_sample_count = class_sample_count / len(label)
        weight = 1 / class_sample_count
        """
        return samples_weight

    

def get_loaders(args, train_path, valid_path):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.pre_train_model, return_attention_mask=True)
    data_collator = DataCollatorCTCWithPadding(feature_extractor=feature_extractor, padding=True, max_length=args.max_length)

    train_dataset = MERDataset(train_path)
    class_weight = train_dataset.class_weight_k()
    valid_dataset = MERDataset(valid_path)
    sampler = WeightedRandomSampler(weights=class_weight, num_samples=train_dataset.__len__())
    # sampler = DistributedProxySampler(
    # ExhaustiveWeightedRandomSampler(class_weight, num_samples=train_dataset.__len__()))
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size, sampler=sampler, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=data_collator)


    return train_dataloader, valid_dataloader, class_weight

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
    def __init__(self, args, config, dropout=0.3, mask_feature_prob=0.05, mask_time_prob=0.08, mask_time_length=15, mask_feature_length=64, pooling_mode="mean"):
        super().__init__(config)
        self.hubert = HubertModel.from_pretrained(args.pre_train_model)
        self.hubert.encoder.gradient_checkpointing = False
        # self.hubert.config.mask_feature_prob = mask_feature_prob
        # self.hubert.config.mask_time_prob = mask_time_prob
        # self.hubert.config.mask_time_length = mask_time_length
        # self.hubert.config.mask_feature_length = mask_feature_length

        self.dropout = nn.Dropout(self.hubert.config.final_dropout)

        self.pooling_mode = pooling_mode

        self.classifier = HubertClassificationHead(dropout=dropout)
    
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

class CELoss(nn.Module):

    def __init__(self, weight=None):
        super(CELoss, self).__init__()
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

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

def overall_metric(emo_fscore, val_mse):
    final_score = emo_fscore - val_mse * 0.25
    return final_score

########################################################
########### main training/testing function #############
########################################################
def train_model(accelerator, model, cls_loss, dataloader, optimizer=None, train=False):
    
    emo_probs, emo_labels = [], []

    assert not train or optimizer!=None
    
    model.train()

    for data in tqdm(dataloader):
        ## analyze dataloader
        input_values, attention_mask, emos = data["input_values"], data["attention_mask"], data["emo_labels"]
        
        ## add cuda
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            emos_out = model(input_values=input_values, attention_mask=attention_mask)
            loss = cls_loss(emos_out, emos)
            accelerator.backward(loss)
            optimizer.step()

        all_emos_out, all_emos = accelerator.gather_for_metrics((emos_out, emos))
        emo_probs.append(all_emos_out.data.cpu().numpy())
        emo_labels.append(all_emos.data.cpu().numpy())

    ## evaluate on discrete labels
    emo_probs  = np.concatenate(emo_probs)
    emo_labels = np.concatenate(emo_labels)
    emo_preds = np.argmax(emo_probs, 1)
    emo_accuracy = accuracy_score(emo_labels, emo_preds)
    emo_fscore = f1_score(emo_labels, emo_preds, average='weighted')

    ## evaluate on dimensional labels

    save_results = {}
    # item1: statistic results
    save_results['emo_fscore'] = emo_fscore
    save_results['emo_accuracy'] = emo_accuracy
    # item2: sample-level results
    save_results['emo_probs'] = emo_probs
    save_results['emo_labels'] = emo_labels
    # item3: latent embeddings
    return save_results

def eval_model(accelerator, model, cls_loss, dataloader, optimizer=None, train=False):
    
    emo_probs, emo_labels = [], []
    
    model.eval()

    for data in tqdm(dataloader):
        ## analyze dataloader
        input_values, attention_mask, emos = data["input_values"], data["attention_mask"], data["emo_labels"]

        with accelerator.autocast():
            with torch.no_grad():
                emos_out = model(input_values=input_values, attention_mask=attention_mask)

        all_emos_out,  all_emos = accelerator.gather_for_metrics((emos_out, emos))

        emo_probs.append(all_emos_out.data.cpu().numpy())
        emo_labels.append(all_emos.data.cpu().numpy())

    ## evaluate on discrete labels
    emo_probs  = np.concatenate(emo_probs)
    emo_labels = np.concatenate(emo_labels)
    emo_preds = np.argmax(emo_probs, 1)
    emo_accuracy = accuracy_score(emo_labels, emo_preds)
    emo_fscore = f1_score(emo_labels, emo_preds, average='weighted')

    ## evaluate on dimensional labels

    save_results = {}
    # item1: statistic results
    save_results['emo_fscore'] = emo_fscore
    save_results['emo_accuracy'] = emo_accuracy
    # item2: sample-level results
    save_results['emo_probs'] = emo_probs
    save_results['emo_labels'] = emo_labels

    return save_results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true', default=False, help='whether use debug to limit samples')
    parser.add_argument('--save_root', type=str, default='./new_train', help='save prediction results and models')

    ## Params for model
    parser.add_argument('--n_classes', type=int, default=7, help='number of classes [defined by args.label_path]')
    parser.add_argument('--pooling_model', type=str, default="mean", help="method for aggregating frame-level into utterence-level")

    ## Params for training
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=1, metavar='BS', help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, metavar='nw', help='number of workers')
    parser.add_argument('--epochs', type=int, default=50, metavar='E', help='number of epochs')
    parser.add_argument('--seed', type=int, default=100, help='make split manner is same with same seed')
    parser.add_argument('--fp16', type=bool, default=True, help='whether to use fp16')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='gradient_accumulation_steps')
    parser.add_argument('--max_length', type=int, default=10, help='max length of audio')
    parser.add_argument('--train_src', type=str, default=2, help='train_src_path')
    parser.add_argument('--valid_src', type=str, default=10, help='valid src path')
    parser.add_argument('--pre_train_model', type=str, default=10, help='pre-trian model path')
    
    args = parser.parse_args()

    train_src_path = args.train_src
    valid_src_path = args.valid_src
    model_path = args.save_root
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    print (f'====== Reading Data =======')
    # train_loaders, eval_loaders, test_loaders, adim, tdim, vdim = get_loaders(args, config) 
    train_loader, eval_loader, class_weight = get_loaders(args, train_src_path, valid_src_path)      
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision = 'fp16',
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
        )
    max_eval_metric = -100
    print (f'====== Training and Evaluation =======')

    start_time = time.time()
    name_time  = time.time()

    print (f'Step1: build model (each folder has its own model)')
    config = AutoConfig.from_pretrained(args.pre_train_model)
    model = HubertForClassification(args, config, dropout=args.dropout, pooling_mode=args.pooling_model)
    model.freeze_feature_extractor()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    model, optimizer, train_loader, eval_loader = accelerator.prepare(model, optimizer, train_loader, eval_loader)
    device = accelerator.device
    class_weight = class_weight.to(device)
    cls_loss = torch.nn.CrossEntropyLoss()

    max_eval_metric = -100

    print (f'Step2: training (multiple epoches)')

    eval_fscores = []

    for epoch in range(args.epochs):

        ## training and validation
        train_results = train_model(accelerator, model, cls_loss, train_loader, optimizer=optimizer, train=True)
        eval_results  = eval_model(accelerator, model, cls_loss, eval_loader,  optimizer=None,      train=False)
        
        # eval_fscores.append(eval_results['emo_fscore'])
        if accelerator.is_main_process:
            print ('epoch:%d; train_fscore:%.4f; eval_fscore:%.4f' %(epoch+1, train_results['emo_fscore'], eval_results['emo_fscore']))

        if max_eval_metric < eval_results['emo_fscore']:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            milestone = model_path + "/" + "fine_tune_wav_model" + str(ii)
            unwrapped_model.save_pretrained(milestone, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
            max_eval_metric = eval_results['emo_fscore']
            




    






