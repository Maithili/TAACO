import json
import numpy as np
from sklearn.metrics import confusion_matrix

def combine_confusion_matrix_dicts(list_of_cms):
    if len(list_of_cms) == 0: return None
    cm = ConfusionMatrix(list_of_cms[0].keys())
    for cm_dict in list_of_cms:
        cm.add_cm(ConfusionMatrix.from_dict(cm_dict))
    return cm.get_numbers_dict()

class ConfusionMatrix:
    def __init__(self, classes):
        self.classes = classes
        self.cm_dict = {c1:{c2:0 for c2 in classes} for c1 in classes}
        self.cm_array = np.zeros((len(self.classes), len(self.classes)), dtype=int)
    
    @staticmethod
    def from_dict(cm_dict, classes=None):
        if classes is None:
            classes = list(cm_dict.keys())
        cm_array = np.array([[cm_dict[c1][c2] for c2 in classes] for c1 in classes])
        cm_obj = ConfusionMatrix(classes)
        cm_obj.cm_array = cm_array
        cm_obj.cm_dict = cm_dict
        return cm_obj

    def precision(self):
        arr = self.cm_array.diagonal()/np.clip(self.cm_array.sum(axis=0), a_min=1e-8, a_max=None)
        return {c:arr[i] for i,c in enumerate(self.classes)}

    def recall(self):
        arr = self.cm_array.diagonal()/np.clip(self.cm_array.sum(axis=1), a_min=1e-8, a_max=None)
        return {c:arr[i] for i,c in enumerate(self.classes)}
        
    def f1_macro(self):
        p = self.precision()
        r = self.recall()
        return sum([2*p[c]*r[c]/max(p[c]+r[c], 1e-8) for c in self.classes])/len(self.classes)

    def accuracy(self):
        return self.cm_array.diagonal().sum()/max(self.cm_array.sum(), 1e-8)
        
    def add_data(self, data):
        preds = [d[0] for d in data]
        golds = [d[1] for d in data]
        self.cm_array += confusion_matrix(golds, preds, labels=self.classes)
        self.cm_dict = {c1:{c2:int(self.cm_array[i,j]) for j,c2 in enumerate(self.classes)} for i,c1 in enumerate(self.classes)}

    def add_cm(self, cm_other):
        if isinstance(cm_other, ConfusionMatrix):
            cm_other = cm_other.cm_array
        self.cm_array += cm_other
        self.cm_dict = {c1:{c2:int(self.cm_array[i,j]) for j,c2 in enumerate(self.classes)} for i,c1 in enumerate(self.classes)}

    def str(self):
        json.dumps(self.cm_dict)
        
    def get_numbers_dict(self):
        return {'accuracy':self.accuracy(), 'precision':self.precision(), 'recall':self.recall(), 'confusion_matrix':self.cm_dict}

def _cm_dict_to_thing_not_done(cm_dict):
    sum = 0
    for gt in cm_dict:
        if gt == 'no_action': continue
        for pred in cm_dict[gt]:
            if pred == 'no_action':
                sum += cm_dict[gt][pred]
    return sum

def _cm_dict_to_robot_crossed_boundaries(cm_dict):
    sum = 0
    for gt in cm_dict:
        if gt in ['no_action','remind']:
            for pred in cm_dict[gt]:
                if pred in ['do_now','do_later']:
                    sum += cm_dict[gt][pred]
    return sum

def _cm_dict_to_unnecessary_interaction(cm_dict):
    sum = 0
    for gt in cm_dict:
        if gt == 'remind': continue
        for pred in cm_dict[gt]:
            if pred == 'remind':
                sum += cm_dict[gt][pred]
    return sum

def _cm_dict_to_delay_or_disturbance(cm_dict):
    sum = 0
    for gt in cm_dict:
        if gt in ['do_now','do_later']:
            for pred in cm_dict[gt]:
                if pred in ['do_now','do_later']:
                    if gt != pred:
                        sum += cm_dict[gt][pred]
    return sum

def _cm_dict_oll_korrect(cm_dict):
    sum = 0
    for gt in cm_dict:
        for pred in cm_dict[gt]:
            if gt == pred:
                sum += cm_dict[gt][pred]
    return sum

def cm_dict_to_subjective(cm_dict):
    result = {
        'correct':_cm_dict_oll_korrect(cm_dict),
        'unnecessary_interaction':_cm_dict_to_unnecessary_interaction(cm_dict), 
        'delay_or_disturbance':_cm_dict_to_delay_or_disturbance(cm_dict), 
        'task_not_done':_cm_dict_to_thing_not_done(cm_dict), 
        'performed_prohibioted_task':_cm_dict_to_robot_crossed_boundaries(cm_dict), 
        }
    
    sum_all_res = sum(result.values())    
    result['total'] = sum_all_res
    
    return result

def complete_evals(episodes, do_now_in_outputs=False):
    if do_now_in_outputs:
        result = {'accuracy':None, 'accuracy_wo_do_now':None,'num_episodes':None, 'num_episodes_wo_do_now':None,'confusion_matrix_obj':None,'episodes':episodes}
    else:
        result = {'accuracy':None, 'num_episodes':None, 'confusion_matrix_obj':None,'episodes':episodes}
    result['num_episodes'] = max(1e-8, len([r for r in result['episodes'] if r[1] is not None]))
    result['accuracy'] = sum([1 if r[0] == r[1] and r[1] is not None else 0 for r in result['episodes']])/result['num_episodes']
    if do_now_in_outputs:
        result['num_episodes_wo_do_now'] = max(1e-8, len([r for r in result['episodes'] if r[1] is not None and r[1] != 'do_now']))
        result['accuracy_wo_do_now'] = sum([1 if r[0] == r[1] and r[1] is not None and r[1] != 'do_now' else 0 for r in result['episodes']])/result['num_episodes_wo_do_now']
    classes = list(set([r[1] for r in result['episodes']]).union(set([r[0] for r in result['episodes']])))
    if all([c in [0,1] for c in classes]): classes = [0,1]
    elif all([c in ['do_now','do_later','remind','no_action'] for c in classes]): classes = ['do_now','do_later','remind','no_action']
    elif all([c in ['null','do_now','do_later','remind','no_action'] for c in classes]): classes = ['null','do_now','do_later','remind','no_action']
    else: raise Exception(f"Unknown classes {classes}")
    cm = ConfusionMatrix(classes)
    cm.add_data(result['episodes'])
    result['confusion_matrix_obj'] = cm
    return result
