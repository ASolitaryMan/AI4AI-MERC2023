from sklearn.utils.class_weight import compute_class_weight
from transformers import BertPreTrainedModel, BertConfig, BertModel, BertForSequenceClassification
import numpy as np
import torch
from transformers import BertTokenizer
import random
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
import random
import torch.nn as nn
from accelerate import Accelerator
import time
import argparse
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import mean_squared_error
from accelerate import DistributedDataParallelKwargs
from tqdm import tqdm
import os
from torch.utils.data import WeightedRandomSampler

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

    feature_extractor: BertTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List) -> Dict[str, torch.Tensor]:
        input_features, emo_label = [], []

        for feature in features:
            input_features.append(feature[0])
            emo_label.append(feature[1])
        
        batch = self.feature_extractor(
            text=input_features,
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            # max_length=max_len,                # Max length to truncate/pad
            padding=self.padding,          # Pad sentence to max length
            # truncation=True,
            return_attention_mask=True,
            return_tensors="pt" 
        )

        d_type = torch.long if isinstance(emo_label[0], int) else torch.float32
        batch["emo_labels"] = torch.tensor(emo_label, dtype=d_type)

        return batch

class TextForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.cls = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.init_weights()
    
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        emo = self.cls(pooled_output)


        return emo


class MERDataset(Dataset):

    def __init__(self, src_path):
        all_lines = read_text(src_path)
        self.label = []
        self.wav_list = []
        for line in all_lines:
            tmp = line.strip().split("\n")[0].split()
            self.wav_list.append(tmp[2])
            self.label.append(emo2idx[tmp[-1]])
        
    def __getitem__(self, index):

        text = self.wav_list[index]
        lab = self.label[index]
    
        return text, lab

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
    feature_extractor = BertTokenizer.from_pretrained("/home/lqf/workspace/Merworkshop/MER2023-Baseline/tools/transformers/chinese-macbert-large")
    data_collator = DataCollatorCTCWithPadding(feature_extractor=feature_extractor, padding=True)

    train_dataset = MERDataset(train_path)
    class_weight = train_dataset.class_weight_k()
    valid_dataset = MERDataset(valid_path)
    sampler = WeightedRandomSampler(weights=class_weight, num_samples=train_dataset.__len__())
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size, sampler=sampler, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=data_collator)
    

    return train_dataloader, valid_dataloader, class_weight

def overall_metric(emo_fscore, val_mse):
    final_score = emo_fscore - val_mse * 0.25
    return final_score

########################################################
########### main training/testing function #############
########################################################

def train_model(accelerator, model, cls_loss, dataloader, optimizer=None, train=False):

    assert not train or optimizer!=None
    
    model.train()
    batch_losses = []

    for data in tqdm(dataloader):
        ## analyze dataloader
        input_values, attention_mask, emos, token_type_ids = data["input_ids"], data["attention_mask"], data["emo_labels"], data["token_type_ids"]
        
        ## add cuda
        with accelerator.accumulate(model):
        # with accelerator.autocast():
            optimizer.zero_grad()
            emos_out = model(input_ids=input_values, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = cls_loss(emos_out, emos)
            accelerator.backward(loss)
            optimizer.step()
            batch_losses.append(loss.item())

    return np.array(batch_losses).mean()

def eval_model(accelerator, model, eval_dataloader):
    emo_probs, emo_labels = [], []
    
    model.eval()

    for data in tqdm(eval_dataloader):
        ## analyze dataloader
        input_values, attention_mask, emos, token_type_ids = data["input_ids"], data["attention_mask"], data["emo_labels"], data["token_type_ids"]

        with accelerator.autocast():
            with torch.no_grad():
                emos_out = model(input_ids=input_values, attention_mask=attention_mask, token_type_ids=token_type_ids)

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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true', default=False, help='whether use debug to limit samples')
    parser.add_argument('--save_root', type=str, default='./saved_text_x', help='save prediction results and models')
    parser.add_argument('--savewhole', action='store_true', default=False, help='whether save latent embeddings')

    ## Params for model
    parser.add_argument('--n_classes', type=int, default=7, help='number of classes [defined by args.label_path]')
    parser.add_argument('--num_folder', type=int, default=5, help='folders for cross-validation [defined by args.dataset]')
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
    parser.add_argument('--max_length', type=int, default=6, help='max length of audio')
    
    args = parser.parse_args()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision = 'fp16',
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
        )
    
    setup_seed(args.seed)

    train_src_path = "/home/lqf/workspace/MERC2023workspace/new_train.scp"
    valid_src_path = "/home/lqf/workspace/MERC2023workspace/val.scp"
    model_path = args.save_root
    
    if accelerator.is_main_process:
        if not os.path.exists(model_path):
            os.makedirs(model_path)

    accelerator.print (f'====== Reading Data =======')
    train_loader, eval_loader, class_weight = get_loaders(args, train_src_path, valid_src_path)      

    max_eval_metric = -100
    accelerator.print (f'====== Training and Evaluation =======')

    xconfig = BertConfig.from_pretrained("/home/lqf/workspace/Merworkshop/MER2023-Baseline/tools/transformers/chinese-macbert-large", num_labels=args.n_classes, hidden_dropout_prob=args.dropout)
    model = TextForSequenceClassification.from_pretrained("/home/lqf/workspace/Merworkshop/MER2023-Baseline/tools/transformers/chinese-macbert-large", config=xconfig)

    reg_loss = torch.nn.MSELoss()
    cls_loss = torch.nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    model, optimizer, train_loader, eval_loader = accelerator.prepare(model, optimizer, train_loader, eval_loader)
    max_eval_metric = -100

    accelerator.print (f'Step2: training (multiple epoches)')

    for epoch in range(args.epochs):

        train_results = train_model(accelerator, model, cls_loss, train_loader, optimizer=optimizer, train=True)
        eval_results  = eval_model(accelerator, model, eval_loader)

        if accelerator.is_main_process:
            print ('epoch:%d; train_loss:%.4f; eval_fscore:%.4f' %(epoch+1, train_results, eval_results['emo_fscore']))

        if max_eval_metric < eval_results['emo_fscore']:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            milestone = model_path + "/" + "fine_tune_text_model" + str(ii) 
            unwrapped_model.save_pretrained(milestone, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
            max_eval_metric = eval_results['emo_fscore']