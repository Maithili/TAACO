import sys
sys.path.append('utils')
import os
import shutil
import json
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch

from utils.eval_helpers import ConfusionMatrix, cm_dict_to_subjective

VERBOSE = True

class BaselineAssistance():
    def __init__(self, datadir, persona, logs_dir, num_train_actions):
        os.makedirs(logs_dir, exist_ok=True)
        self.datadir = datadir
        self.persona = persona
        self.logs_dir = logs_dir
        self.confusion_matrix = None
        self.list_of_cms = []
        self.num_train_actions = num_train_actions
    
    def get_train_test_preferences(self):
        preferences_all =  [json.loads(l.strip()) for l in open(os.path.join(self.datadir, '..', f'preferences.jsonl')).readlines()]

        if self.num_train_actions < 0: self.num_train_actions = 1000000
        train_set_data = [json.loads(l.strip()) for l in open(os.path.join(self.datadir, f'preferences_train_expanded.jsonl')).readlines()]
        actions_train = set([d['action'][0] for d in train_set_data if d['action_index'] < self.num_train_actions])

        preferences_train = [p for p in preferences_all if p['action'] in actions_train]
        preferences_test = [json.loads(l.strip()) for l in open(os.path.join(self.datadir, f'preferences_test_gpt.jsonl')).readlines()]

        print(f"Loaded {len(preferences_train)} training preferences and {len(preferences_test)} test preferences with context variations")

        return preferences_train, preferences_test
    
    def _key_from_text(self, datapoint):
        key = ''
        key += datapoint['action'][0]+'_'
        key += '_'.join(datapoint['context'])+'_'
        key = key.replace(' ','_')
        return key

    
    def update_confusion_matrix(self, cm, mrrs):
        if os.path.exists(os.path.join(self.logs_dir,'confusion_matrix.pt')):
            self.confusion_matrix = torch.load(os.path.join(self.logs_dir,'confusion_matrix.pt'))
        if os.path.exists(os.path.join(self.logs_dir,'confusion_matrix.json')):
            self.list_of_cms = json.load(open(os.path.join(self.logs_dir,'confusion_matrix.json'),'r'))['history']
        self.list_of_cms.append(deepcopy(cm.get_numbers_dict()))
        self.list_of_cms[-1]["MRR"] = mrrs['overall']
        self.list_of_cms[-1]["MRR_correct"] = mrrs['correct']
        self.list_of_cms[-1]["MRR_wrong"] = mrrs['wrong']
        if self.confusion_matrix is None:
            self.confusion_matrix = cm
        else:
            self.confusion_matrix.add_cm(cm)
        result_now = {
            'final':{
                'accuracy':self.confusion_matrix.accuracy(),
                'precision':self.confusion_matrix.precision(),
                'recall':self.confusion_matrix.recall(),
                'mrr': np.average([r['MRR'][0] for r in self.list_of_cms], weights=[r['MRR'][1] for r in self.list_of_cms]),
                'mrr_correct': np.average([r['MRR_correct'][0] for r in self.list_of_cms], weights=[r['MRR_correct'][1] for r in self.list_of_cms]),
                'mrr_wrong': np.average([r['MRR_wrong'][0] for r in self.list_of_cms], weights=[r['MRR_wrong'][1] for r in self.list_of_cms]),
                'confusion_matrix':self.confusion_matrix.cm_dict
            },
            'history':self.list_of_cms
            }
        torch.save(self.confusion_matrix, os.path.join(self.logs_dir,'confusion_matrix.pt'))
        json.dump(result_now, open(os.path.join(self.logs_dir,'confusion_matrix.json'),'w'), indent=4)
        
    def postprocess_results(self, result):
        result_all = {
            'accuracy': [r['accuracy'] for r in result['history']],
            'mrr_attn_based': [0 for r in result['history']],
            'subj': [cm_dict_to_subjective(r['confusion_matrix']) for r in result['history']]
        }
        return result_all