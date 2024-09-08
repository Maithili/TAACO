import sys
sys.path.append('dataloader')
from copy import deepcopy
import json
import time
import os
import random
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from embedders import get_embedder


class DataSplit():
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert idx < len(self.data), f"Index {idx} out of bounds for dataset of size {len(self.data)}"
        return self.data[idx]

class ConceptActionLabelledDataset():
    def __init__(self, filepath, embedding_path, concept_list=None, **kwargs):
        self.config = kwargs
        data = []
        sampler_weights = []
        for path, weight in filepath['train'].items():
            num_data = len(open(path).readlines())
            if num_data == 0:
                print(f"WARNING: {path} is empty!")
                continue
            data += [json.loads(l.strip()) for l in open(path).readlines()]
            sampler_weights += [1/num_data * weight] * num_data
            assert 'gen' in path or 'action_idx' in json.loads(open(path).readlines()[0]), "All non-generated match data must have an action_idx"
        data = [d for d in data if 'action_idx' not in d or d['action_idx'] < self.config['num_train_actions'] or self.config['num_train_actions'] < 0]
        
        data_test = []
        if len(filepath['test']) == 0:
            dataidx = int(round(len(data)* self.config['test_fraction']))
            data_test = deepcopy(data[:dataidx])
            data = deepcopy(data[dataidx:])
            sampler_weights = deepcopy(sampler_weights[dataidx:])
        else:
            for path in filepath['test']:
                data_test += [json.loads(l.strip()) for l in open(path).readlines()]
        
        items = list(set([d['item'] for d in data]))
        test_items = list(set([d['item'] for d in data_test]))
        concepts = list(set([d['concept'] for d in data]))
        if concept_list is not None:
            concepts = concept_list
            data_idxs = [i for i,d in enumerate(data) if d['concept'] in concepts]
            data = [d for d in data if d['concept'] in concepts]
            sampler_weights = [sampler_weights[i] for i in data_idxs]
            data_test = [d for d in data_test if d['concept'] in concepts]
        self.data_train = data[int(round(len(data)* self.config['val_fraction_concepts'])):]
        self.data_val = data[:int(round(len(data)* self.config['val_fraction_concepts']))]
        self.sampler_weights_val = sampler_weights[:int(round(len(data)* self.config['val_fraction_concepts']))]
        self.sampler_weights_train = sampler_weights[int(round(len(data)* self.config['val_fraction_concepts'])):]
        self.data_test = data_test

        print(f"Dataset Loaded:")
        print(f"     --> {len(self.data_train)} Train set")
        print(f"     --> {len(self.data_test)} Test set")
        print(f"     --> {len(self.data_val)} Val set")
        print(f"     --> {len(concepts)} Concepts")

        if os.path.exists(embedding_path): 
            run = random.randint(0,1000)
            for trial in range(10):
                try:
                    embedding_model = get_embedder(self.config['embedder'], path=embedding_path)
                    break
                except:
                    print(f"{run}: Failed to load embedding model, retrying...")
                    time.sleep(random.random()*2)
                if trial == 9:
                    embedding_model = get_embedder(self.config['embedder'])
                    break
        else: 
            embedding_model = get_embedder(self.config['embedder'])
        embedding_model.add_items(items+test_items)
        embedding_model.add_concepts(concepts)
        self.embedding_map = embedding_model.map
        embedding_model.save(embedding_path)

        self.concepts = {k:self.embedding_map['concepts'][k] for k in concepts}

        self.input_embed_dim = list(self.embedding_map['items'].values())[0].shape[1]
        print(f"Using embedding map {self.config['embedder']} which has dimension {self.input_embed_dim}")
        
        def rescale_score(score):
            return (score-0.5)/10
        
        self.collate_fn = lambda data : {'items':torch.cat([self.embedding_map['items'][datum['item'].lower()] for datum in data], dim=0),
                                         'concepts':torch.cat([self.embedding_map['concepts'][datum['concept'].lower()] for datum in data], dim=0),
                                         'labels':torch.stack([torch.tensor(rescale_score(int(datum['match']))) for datum in data], dim=0),
                                         'items_text':[datum['item'] for datum in data],
                                         'concepts_text':[datum['concept'] for datum in data],}

    def has_data(self, type_data='all'):
        if type_data == 'all':
            return len(self.data_train) > 0 or len(self.data_val) > 0 or len(self.data_test) > 0
        if type_data == 'train':
            return len(self.data_train) > 0 or len(self.data_val) > 0
        if type_data == 'test':
            return len(self.data_test) > 0

    def get_train_loader(self):
        return DataLoader(DataSplit(self.data_train), 
                          num_workers=1, 
                          batch_size=self.config['batch_size_concept'], 
                          collate_fn=self.collate_fn,
                          sampler=WeightedRandomSampler(self.sampler_weights_train, len(self.sampler_weights_train)))
    
    def get_val_loader(self):
        return DataLoader(DataSplit(self.data_val), 
                          num_workers=1, 
                          batch_size=self.config['batch_size_concept'], 
                          collate_fn=self.collate_fn,
                          sampler=WeightedRandomSampler(self.sampler_weights_val, len(self.sampler_weights_val)))

    def get_test_loader(self):
        return DataLoader(DataSplit(self.data_test), 
                          num_workers=1, 
                          batch_size=self.config['batch_size_concept'], 
                          collate_fn=self.collate_fn,
                          shuffle=False)
