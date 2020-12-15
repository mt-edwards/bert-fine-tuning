# import dependencies
import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# load data
df = pd.read_csv("data/cola_public/raw/in_domain_train.tsv",
                 delimiter='\t',
                 header=None,
                 names=['sentence_source', 'label', 'label_notes', 'sentence'])

# sentences and labels
sentences = df.sentence.values
labels = df.label.values

# BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# BERT tokenization
input_ids = []
attention_masks = []

for sent in sentences:
    encoded_dict = tokenizer.encode_plus(sent,
                                         padding='max_length',
                                         max_length=64,
                                         return_tensors='pt',
                                         return_attention_mask=True)
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

# convert lists to pytorch tensors
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# training and validation split
dataset = TensorDataset(input_ids, attention_masks, labels)
train_size = int(0.9 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

# batch size
batch_size = 32

# data loaders
train_dataloader = DataLoader(train_dataset,
                              sampler = RandomSampler(train_dataset),
                              batch_size = batch_size)
valid_dataloader = DataLoader(valid_dataset,
                              sampler = SequentialSampler(valid_dataset),
                              batch_size = batch_size)
