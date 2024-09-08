import json
import os
import random
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.embedders import get_embedder
from model.ConceptActionMappingModule import ConceptActionMappingModule

class DataSplit():
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

conflicting_contexts = [
    ["early in the morning"],
    ["user is asleep", "guests are present"],
    ["user is asleep", "user is in a rush"],
    ["weekday","weekend"]
]
def all_context_combinations(list_of_contexts):
    all_context_combinations = [[]]
    for context_var in list_of_contexts:
        this_conflicts = []
        for conflict in conflicting_contexts:
            if context_var in conflict:
                this_conflicts += [c for c in conflict if c != context_var]
        new_context_combinations = []
        for comb_so_far in all_context_combinations:
            if any([c in comb_so_far for c in this_conflicts]):
                continue
            new_context_combinations.append(comb_so_far+[context_var])
        all_context_combinations += new_context_combinations
    all_context_combinations_binary = []
    for comb in all_context_combinations:
        binary = [1 if c in comb else 0 for c in list_of_contexts]
        all_context_combinations_binary.append(binary)
    return all_context_combinations_binary

class AssistanceActionDataset():
    def __init__(self, dirpath, concept_src='model', new_concept_files_dir=None, no_validation=False, **kwargs):
        self.config = kwargs
        self.config['concept_src'] = concept_src

        print(f"********** Reading preferences from {os.path.join(dirpath, f'preferences_train/test_expanded.jsonl')} **********")
        print(f"USING {concept_src.upper()} CONCEPT MAPS!!")

        data_in_train = [json.loads(l.strip()) for l in open(os.path.join(dirpath, f'preferences_train_expanded.jsonl')).readlines()]
        data_in_test = [json.loads(l.strip()) for l in open(os.path.join(dirpath, f'preferences_test_from_gpt.jsonl')).readlines()]
        data_out_test = []
        
        self.persona = self.config['persona']
        self.breakdowns = json.load(open(os.path.join(dirpath, 'all_concepts.json')))
        self.all_context_combinations = all_context_combinations(self.breakdowns['context'])
        
        name_to_concept = {}
        name_to_concept_new = {}
        self.dirpath = dirpath
        all_items = {}
        for space in ['action','activity','location','object']:
            all_items[space] = set()
            for d in data_in_train+data_in_test:
                all_items[space].update(d[space])
            
        def organize_concepts_from_files(filenames, space, init_dict={}, verbose=False):
            file_data = []
            for f in filenames:
                file_data += [json.loads(l) for l in open(f).readlines()]
            file_data_organized = {}
            file_data_organized.update(init_dict)
            for data in file_data:
                if data['item'].lower() not in all_items[space]: continue
                if data['concept'].lower() not in self.breakdowns['input'][space]: continue
                match_norm = (data['match']-0.5)/10
                match_norm = min(1, max(0, match_norm))
                assert match_norm <= 1 and match_norm >= 0, f"Match norm is not between 0 and 1: {match_norm}"
                if verbose:
                    print(f"Found {data['item']} to match {data['concept']}: {match_norm}")
                if len(init_dict) > 0:
                    if data['item'].lower() not in init_dict or data['concept'].lower() not in init_dict[data['item'].lower()]:
                        import pdb; pdb.set_trace()
                if data['item'].lower() not in file_data_organized:
                    file_data_organized[data['item'].lower()] = {}
                file_data_organized[data['item'].lower()][data['concept'].lower()] = match_norm
            num_concepts_local = None
            for item, item_data in file_data_organized.items():
                assert num_concepts_local is None or len(item_data.keys()) == num_concepts_local, f"Number of concepts for {item} is not consistent: {len(item_data.keys())} vs {num_concepts_local}"
                num_concepts_local = len(item_data.keys())
            return file_data_organized

        self.embedding_maps = {}
        concept_spaces_used = [k for k in ['action','activity','location','object'] if len(self.breakdowns['input'][k]) > 0]
        for k in concept_spaces_used:
            embedding_model = get_embedder(self.config['embedder'], path=os.path.join(dirpath, '../..', f'gen_embedding_map_{k}.pt'))
            embedding_model.add_items(list(all_items[k]))
            self.embedding_maps[k] = embedding_model.map['items']
            self.embedding_maps[k] = {item.lower():bertem.detach().to('cpu') for item,bertem in self.embedding_maps[k].items() if item.lower() in all_items[k]}

        for k in concept_spaces_used:
            if concept_src == 'model':
                embedding_keys = list(self.embedding_maps[k].keys())
                embedding_values = torch.cat(list(self.embedding_maps[k].values()), dim=0)
                concept_model = ConceptActionMappingModule.load_from(os.path.join(self.config['logs_dir'], f'concept_{k}'), concepts_text=self.breakdowns['input'][k])
                concept_values = concept_model.get_item_concept_similarities(items=embedding_values)
                name_to_concept[k] = {name:value.view(-1).detach().to('cpu') for name,value in zip(embedding_keys, concept_values)}
                concept_model_new = ConceptActionMappingModule.load_from(os.path.join(new_concept_files_dir, f'concept_{k}'), no_vocab=True)
                concept_model_new.add_concept_vocab(concept_model.concept_vocab)
                concept_model_new.cfg['num_concepts_overall'] = concept_model.cfg['num_concepts_overall']
                concept_values_new = concept_model_new.get_item_concept_similarities(items=embedding_values)
                name_to_concept_new[k] = {name:value.view(-1).detach().to('cpu') for name,value in zip(embedding_keys, concept_values_new)}
            if concept_src == 'gpt' or concept_src == 'no_concepts':
                gpt_data_organized = organize_concepts_from_files([f'data/all_concept_map_{k}_gpt4.jsonl'], k)
                name_to_concept[k] = {it:torch.tensor([gpt_data_organized[it.lower()][c.lower()] for c in self.breakdowns['input'][k] if c in gpt_data_organized[it].keys()]) for it in gpt_data_organized.keys()}
                name_to_concept_new[k] = name_to_concept[k]
            if concept_src == 'user':
                gpt_data_organized = organize_concepts_from_files([f'data/all_concept_map_{k}_gpt4.jsonl'], k)
                user_data_organized = organize_concepts_from_files([os.path.join(dirpath, f'train_user_concept_map_{k}.jsonl'), os.path.join(dirpath, f'test_user_concept_map_{k}.jsonl')], k, init_dict=gpt_data_organized)
                name_to_concept[k] = {it:torch.tensor([user_data_organized[it.lower()][c.lower()] for c in self.breakdowns['input'][k] if c in user_data_organized[it].keys()]) for it in user_data_organized.keys()}
        
        for space in concept_spaces_used:
            for item,match in name_to_concept[space].items():
                assert torch.all(match <= 1) and torch.all(match >= 0), f"Match values out of range: {space}-{item}--{match.min().item()} to {match.max().item()}"


        def process_datapoint(d, new=False, freeze_context_sample=False, all_context_samples=False):
            proc = {'input':{},'context':None}
            lang_emb = {}
            proc_text = {'input':{}}
            for k in concept_spaces_used:
                if isinstance(d[k], list):
                    if len(d[k]) > 1:
                        assert k in ['object', 'location']
                else:
                    assert k in ['action', 'activity']
                    d[k] = [d[k]]
                lang_emb[k] = [self.embedding_maps[k][dk.lower()].squeeze(0) for dk in d[k]]
                if new:
                    proc['input'][k] = [name_to_concept_new[k][dk.lower()] for dk in d[k]]
                else:
                    proc['input'][k] = [name_to_concept[k][dk.lower()] for dk in d[k]]
                for idx in range(len(proc['input'][k])):
                    assert isinstance(proc['input'][k][idx], torch.Tensor), f"Input{k}: {proc['input'][0]} not a tensor"
                proc_text['input'][k] = list(d[k])

            def is_context(c):
                if not freeze_context_sample:
                    for con,flag in d['context']:
                        if con==c:
                            return 1 if flag else -1
                    return 0
                else:
                    return 1 if c in d['context'] else -1
            
            proc['context'] = [is_context(c) for c in self.breakdowns['context']]
            possible_contexts = [proc['context']]
            if all_context_samples:
                possible_contexts = [[1 if i==1 else -1 for i in comb] for comb in self._all_context_combination(proc['context'])]
            elif freeze_context_sample:
                proc['context'] = [1 if i==1 else -1 for i in self._sample_context_combination(proc['context'])]
                possible_contexts = [proc['context']]
                data_out_test.append(d)
                data_out_test[-1]['context'] = [c for c,flag in zip(self.breakdowns['context'], proc['context']) if flag==1]
            
            d['preference'] = d['preference'].strip().replace(' ','_')
            proc_text['output'] = d['preference']
            preference = [1 if o==d['preference'] else 0 for o in self.breakdowns['output']]
            assert sum(preference) > 0, f"Preference {d['preference']} not found in {self.breakdowns['output']}"
            
            converted_data = []
            for cont in possible_contexts:
                proc['context'] = cont
                converted_data.append({'encoded':deepcopy(proc),
                                    'input_lm_embeds':lang_emb,
                                    'preference':preference, 
                                    'explanation':d['explanation'], 
                                    'explanation_opposing':d['explanation_opposing'], 
                                    'explanation_context':[x[0] for x in d['explanation_context']],
                                    'text':proc_text,
                                    'num_precedents':d['num_precedents'] if 'num_precedents' in d.keys() else 0,
                                    })

            return converted_data

        val_idx = int(round(len(data_in_train)*self.config['val_fraction_assistance']))
        if no_validation: val_idx = 0
        data_in_val = data_in_train[:val_idx]
        data_in_train = data_in_train[val_idx:]
        
        self.data_train = []
        self.data_val = []
        self.data_test = []
        for d in data_in_train:
            if self.config['num_train_actions'] >= 0 and d['action_index'] >= self.config['num_train_actions']: continue
            self.data_train += process_datapoint(d)
            if concept_src == 'model':
                self.data_train += process_datapoint(d, new=True)
        for d in data_in_val:
            self.data_val += process_datapoint(d)
        for d in data_in_test:
            self.data_test += process_datapoint(d, freeze_context_sample=True)

        if no_validation: self.data_val = self.data_test

        print(f"Dataset Loaded: Persona {self.persona}")
        print(f"     --> {len(self.data_train)} Train Datapoints")
        print(f"     --> {len(self.data_val)} Validation Datapoints")
        print(f"     --> {len(self.data_test)} Test Datapoints")

        open(os.path.join(dirpath, f'preferences_test_gpt.jsonl'),'w').write('\n'.join([json.dumps(d) for d in data_out_test]))

        self.action_vocab = self.breakdowns['output']

    def has_data(self, type_data='all'):
        if type_data == 'all':
            return len(self.data_train) > 0 or len(self.data_test) > 0
        if type_data == 'train':
            return len(self.data_train) > 0
        if type_data == 'test':
            return len(self.data_test) > 0


    def _collate_fn(self, data):
        collated = {
            'input_concepts': {},
            'input_lm_embeds': {},
            'context': {},
            'assistance': torch.tensor([d['preference'] for d in data]),
            'explanation': [],
            'explanation_opposing': [],
            'explanation_context': [],
            'text': [d['text'] for d in data],
            'explanation_based_data': None,
            'explanation_text': [d['explanation'] for d in data],
            'explanation_opposing_text': [d['explanation_opposing'] for d in data],
            'explanation_context_text': [d['explanation_context'] for d in data],
            'num_precedents': torch.tensor([d['num_precedents'] for d in data], dtype=int),
            }
        
        size_items = {space:max([len(d['text']['input'][space]) for d in data]) for space in data[0]['text']['input'].keys()}
        collated['explanation'] = {k:-100*torch.ones((len(data), size_items[k], len(self.breakdowns['input'][k])), dtype=int) for k in data[0]['text']['input'].keys()}
        collated['explanation_opposing'] = {k:-100*torch.ones((len(data), size_items[k], len(self.breakdowns['input'][k])), dtype=int) for k in data[0]['text']['input'].keys()}
        collated['explanation_context'] = torch.zeros((len(data), len(self.breakdowns['context'])), dtype=int)
        for idx_data,d in enumerate(data):
            for exp_key in ['explanation', 'explanation_opposing']:
                for space in d['text']['input'].keys():
                    collated[exp_key][space][idx_data,:len(d['text']['input'][space]),:] = 0
                for explanation in d[exp_key]:
                    space, entity, concept, match = explanation
                    entity_idx = d['text']['input'][space].index(entity)
                    concept_idx = self.breakdowns['input'][space].index(concept)
                    collated[exp_key][space][idx_data][entity_idx][concept_idx] = 1 if match else -1
            for explanation in d['explanation_context']:
                collated['explanation_context'][idx_data][self.breakdowns['context'].index(explanation)] = 1
        
        def combine(lists, different_sizes=True):
            masks = torch.zeros((len(lists), len(lists[0])))
            if different_sizes:
                lengths = torch.tensor([len(l) for l in lists])
                max_length = max(lengths)
                masks = torch.zeros((len(lists), max_length))
                masks[torch.arange(max_length).view(1,-1).repeat(len(lists),1) >= lengths.unsqueeze(-1)] = -float("inf")
                final_size = len(lists), max_length, lists[np.argmax(lengths.numpy())][0].shape[-1]
                lists = [l+[torch.tensor([-1 for _ in range(final_size[-1])]).view(-1)]*(max_length-len(l)) for l in lists]
                if isinstance(lists[0], list):
                    try:
                        tensors = torch.stack([torch.stack(l,dim=0) for l in lists], dim=0)
                        assert tensors.shape == torch.Size(final_size), f"{tensors.shape} == {torch.Size(final_size)}"
                    except Exception as e:
                        print("LIST start")
                        for l in lists:
                            print(l)
                            print()
                        print("LIST end")
                        print(f"FINAL SIZE: {final_size}")
                        print([[ll.shape for ll in l] for l in lists])
                        print(e)
                        assert False
                else:
                    tensors = torch.stack([torch.tensor(l) for l in lists], dim=0)
            else:
                tensors = torch.stack(lists, dim=0)
            return {'features':tensors.float(), 'mask':masks}
        
        for input_space in data[0]['encoded']['input'].keys():
            collated['input_concepts'][input_space] = {}
            assert all([all([[aa<1 for aa in a] for a in d['encoded']['input'][input_space]]) and all([[aa>0 for aa in a] for a in d['encoded']['input'][input_space]]) for d in data]), d['encoded']['input'][input_space]
            collated['input_concepts'][input_space] = combine([d['encoded']['input'][input_space] for d in data])
            assert torch.all(collated['input_concepts'][input_space]['features']<=1), f"Input features out of range: Max. {collated['input_concepts'][input_space]['features'].max().item()}"
            assert torch.all(collated['input_concepts'][input_space]['features']>=-1), f"Input features out of range: Min. {collated['input_concepts'][input_space]['features'].min().item()}"
            collated['input_lm_embeds'][input_space] = combine([d['input_lm_embeds'][input_space] for d in data])
        context_samples = [torch.tensor([self._sample_context_combination(d['encoded']['context'])]).view(-1) for d in data]
        for i,context_sample in enumerate(context_samples):
            collated['text'][i]['context'] = [self.breakdowns['context'][i] for i,c in enumerate(context_sample) if c>0]
        collated['context'] = combine(context_samples, different_sizes=False)

        return collated
    
    def _sample_context_combination(self, mask):
        eligible_combinations = [comb for comb in self.all_context_combinations if all([((c==0 and m==-1) or (c==1 and m==1) or (m==0)) for c,m in zip(comb,mask)])]
        return random.choice(eligible_combinations)
    
    def _all_context_combination(self, mask):
        eligible_combinations = [comb for comb in self.all_context_combinations if all([((c==0 and m==-1) or (c==1 and m==1) or (m==0)) for c,m in zip(comb,mask)])]
        return eligible_combinations
        

    def get_train_loader(self):
        return DataLoader(DataSplit(self.data_train), 
                          num_workers=1, 
                          batch_size=self.config['batch_size_assist'], 
                          collate_fn=self._collate_fn,
                          shuffle=True)

    def get_test_loader(self):
        return DataLoader(DataSplit(self.data_test), 
                          num_workers=1, 
                          batch_size=self.config['batch_size_assist'], 
                          collate_fn=self._collate_fn,
                          shuffle=False)

    def get_val_loader(self):
        return DataLoader(DataSplit(self.data_val), 
                          num_workers=1, 
                          batch_size=self.config['batch_size_assist'], 
                          collate_fn=self._collate_fn,
                          shuffle=False)