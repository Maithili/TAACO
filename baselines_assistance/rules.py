import os
import json
import random
from copy import deepcopy
from tqdm import tqdm

from baselines_assistance.base import BaselineAssistance
from utils.eval_helpers import complete_evals


VERBOSE = True

class PreferenceRules():
    def __init__(self, default_action="do_now") -> None:
        self.rules = []
        self.default_action = default_action

    def _phrase_explanation(self, subrule):
        expl = []
        subrule['items'].sort(key=lambda x: x[0]+x[1])
        if len(subrule['items']) > 1:
            print(f"Subrule has many items!!!!!! {subrule}")
        for idx,item in enumerate(subrule['items']):
            expl.append([item[0], item[1]])
        subrule['context'].sort()
        for context in subrule['context']:
            expl.append([context])
        return expl
            

    def add_preference_datapoint(self, datapoint):
        pref = datapoint['preferences']
        pref[-1]['explanations'].append(["action",datapoint['action'],""])
        explanations_cond = []
        for i,preference_level in enumerate(pref):
            explanations_cond.append([])
            for expl in preference_level['explanations']:
                items = [(expl[0],expl[1])]
                explanations_cond[i].append(items)
            for prev_level in range(i):
                for uncond_expl in explanations_cond[i]:
                    if uncond_expl in explanations_cond[prev_level]:
                        explanations_cond[prev_level].remove(uncond_expl)
        
        rule = []
        for explanations_cond,pref_layer in zip(explanations_cond,pref):
            context = pref_layer['conditions']
            for exp in explanations_cond:
                pref_layer['action'] = pref_layer['action'].replace(' ','_')
                rule.append({'action':pref_layer['action'], 'items':exp, 'context':context})
                if rule not in self.rules:
                    self.rules.append(rule)
    
    def get_preference(self, action, items, context, raw_logs_file):
        desired_action = {}
        for rule in self.rules:
            for subrule in rule:
                if all([i in items for i in subrule['items']]) and (any([c in context for c in subrule['context']]) or len(subrule['context']) == 0):
                    open(raw_logs_file, 'a').write("\n"+f"Found rule {subrule}")
                    if subrule['action'] not in desired_action:
                        desired_action[subrule['action']] = []
                    expl = self._phrase_explanation(subrule)
                    if expl not in desired_action[subrule['action']]:
                        desired_action[subrule['action']].append(expl)
                    break
        if len(desired_action) == 0:
            open(raw_logs_file, 'a').write("\n"+f"WARNING: No rule found for action {action} with items {items} and context {context}")
            return self.default_action, None

        all_actions = {k: v for k, v in sorted(desired_action.items(), key=lambda item: len(item[1]), reverse=True)}
        votes = {k:len(v) for k,v in all_actions.items()}
        
        if len(all_actions) == 1:
            result = list(all_actions.keys())[0]
            open(raw_logs_file, 'a').write("\n"+f"Multiple rules with result {result} for action {action} with items {items} and context {context}")
            return result, all_actions[result]
        else:
            best_actions = [k for k,v in votes.items() if v == max(votes.values())]
            if len(best_actions) != 1:
                random.shuffle(best_actions)
                open(raw_logs_file, 'a').write("\n"+f"Multiple rules, randomly chosen {best_actions[0]} for action {action} with items {items} and context {context}")
            else:
                open(raw_logs_file, 'a').write("\n"+f"Multiple rules with best result {best_actions[0]} for action {action} with items {items} and context {context}")
            return best_actions[0], all_actions[best_actions[0]]

class BaselineAssistance_RuleBased(BaselineAssistance):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def run(self, files_prefix):
        raw_logs_file = os.path.join(self.logs_dir, f'{files_prefix}raw_logs.txt')
        open(raw_logs_file,'w').write('')
        output_file = os.path.join(self.logs_dir, f'{files_prefix}output.json')
        open(output_file, 'w').write('')
        preferences_train, preferences_test = self.get_train_test_preferences()
        preference_rules = PreferenceRules()
        
        for p in preferences_train:
            preference_rules.add_preference_datapoint(p)

        open(raw_logs_file, 'a').write("\n"+"RULES:")
        for r in preference_rules.rules:
            open(raw_logs_file, 'a').write("\n"+str(r))
        open(raw_logs_file, 'a').write("\n\n\n")

        episodes = {}
        for p in tqdm(preferences_test):
            expectation = p['preference'].replace(' ','_') if 'preference' in p else None
            items = []
            items += [("action",d) for d in p['action']]
            items += [("activity",d) for d in p['activity']]
            items += [("object",d) for d in p['object']]
            items += [("location",d) for d in p['location']]
            action, explanations = preference_rules.get_preference(p['action'], items, p['context'], raw_logs_file=raw_logs_file)
            def is_gt_expl(expls, gt_expl):
                if expls is None: return False
                for expl in expls:
                    for gte in gt_expl:
                        if expl[0][0] == gte[0] and expl[0][1] == gte[1]:
                            return True
                    return False
            episodes[self._key_from_text(p)] = {
                'ground_truth': expectation,
                'prediction': action,
                'result': 'correct' if action == expectation else 'wrong',
                'explanation': p['explanation'],
                'explanation_rules': explanations,
                'mrr': 1 if is_gt_expl(explanations, p['explanation']) else 0,
            }
            open(raw_logs_file, 'a').write("\n"+f"{action} (GT){expectation} <-- {str(p)}\n\n")

        result = complete_evals([(r['prediction'], r['ground_truth']) for r in episodes.values()], do_now_in_outputs=True)
        mask_list = [r for r in episodes.values() if len(r['explanation'])>0]
        correct_mask = [r['result']=='correct' for r in mask_list]
        masked_series = [r['mrr'] for r in mask_list]
        result["MRR"] = sum(masked_series)/len(masked_series), len(masked_series)
        masked_series = [r['mrr'] for r,f in zip(mask_list,correct_mask) if f]
        result["MRR_correct"] = sum(masked_series)/len(masked_series), len(masked_series)
        masked_series = [r['mrr'] for r,f in zip(mask_list,correct_mask) if not f]
        result["MRR_wrong"] = sum(masked_series)/len(masked_series), len(masked_series)
        result["episodes"] = episodes

        self.update_confusion_matrix(result['confusion_matrix_obj'], {'overall':result["MRR"], 'correct':result['MRR_correct'], 'wrong':result['MRR_wrong']})

        result_writeable = deepcopy(result)
        result_writeable['confusion_matrix_obj'] = result_writeable['confusion_matrix_obj'].get_numbers_dict()
        open(output_file,'w').write(json.dumps(result_writeable, indent=4))
        print(f"Results written to {output_file}")
