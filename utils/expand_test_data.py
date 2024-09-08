import os
import json

conflicting_contexts = [
    ["early in the morning"],
    ["user is asleep", "guests are present"],
    ["user is asleep", "user is in a rush"],
    ["user is away", "user is nearby"],
    ["user is away", "user is asleep"],
    ["weekday","weekend"],
    ["user is injured or disabled", "user is in a rush"],
    ["adverse weather conditions"]
]

def expand_test_data(persona, fold_num = 0, num_folds = 5):

    fold_str = 'fold_{:02d}'.format(fold_num)

    datadir = os.path.join('data', persona)
    os.makedirs(os.path.join(datadir, fold_str), exist_ok=True)

    preference_file = os.path.join(datadir, f'preferences.jsonl')

    all_preferences = [json.loads(l.strip()) for l in open(preference_file).readlines()]
    concepts = json.load(open(os.path.join(datadir, 'all_concepts.json')))
        
    json.dump(concepts, open(os.path.join(datadir, fold_str, 'all_concepts.json'),'w'), indent=4)

    expanded_preference_file = {'train':os.path.join(datadir, fold_str, 'preferences_train_expanded.jsonl'),
                                'test':os.path.join(datadir, fold_str, 'preferences_test_expanded.jsonl')}
    
    all_sets = {
        'train': {
            'action':{},
            'activity':{},
            'object':{},
            'location':{},},
        'test': {
            'action':{},
            'activity':{},
            'object':{},
            'location':{},}
    }
    
    concept_file = {split:{space:os.path.join(datadir, fold_str, f'{split}_user_concept_map_{space}.jsonl') for space in ['action','activity','object','location']} for split in ['train','test']}

    ## Create random splits
    test_start = fold_num
    step = num_folds
    activity_preferences = {'train': [],
                            'test': all_preferences[test_start::step]}
    activity_preferences['train'] = [p for p in all_preferences if p not in activity_preferences['test']]


    for f in expanded_preference_file.values():
        open(f,'w').write('')

    list_of_explanations = {
        'train':{'action':[],'activity':[],'object':[],'location':[]},
        'test':{'action':[],'activity':[],'object':[],'location':[]}
    }

    num_precedents = {}
    for split in ['train','test']:
        for i_act,activity in enumerate(activity_preferences[split]):
            for space in all_sets[split]:
                if isinstance(activity[space], str):
                    activity[space] = [activity[space]]
                for item in activity[space]:
                    item_low_str = item.lower().strip()
                    if item_low_str not in all_sets[split][space]:
                        all_sets[split][space][item_low_str] = []
                    all_sets[split][space][item_low_str].append(i_act)
            preferences = []
            if 'preferences' in activity:
                preferences = activity['preferences']
            for preference in preferences:
                explanations_completed = []
                for explanation in preference["explanations"]:
                    explanation_full = explanation
                    if len(explanation) == 3: explanation.append(True)
                    explanations_completed.append(explanation_full)
                    list_of_explanations[split][explanation_full[0]].append((explanation_full[1], explanation_full[2], explanation_full[3], i_act))
                preference["explanations"] = explanations_completed
                explanations_completed = []
                for explanation in preference["explanations_opposing"]:
                    explanation_full = explanation
                    if len(explanation) == 3: explanation.append(True)
                    explanations_completed.append(explanation_full)
                    list_of_explanations[split][explanation_full[0]].append((explanation_full[1], explanation_full[2], explanation_full[3], i_act))
                preference["explanations_opposing"] = explanations_completed
            def num_precedents(explanation):
                return [e[1] for e in list_of_explanations['train'][explanation[0]]].count(explanation[2])
            def result(context):
                for p in preferences:
                    if len(p['conditions']) == 0:
                        return p, []
                    overlap = [p for p in p['conditions'] if p in context]
                    if len(overlap) > 0:
                        return p, overlap
                print("WARNING! NO PREFERENCE FOUND")
                raise Exception("No preference found")
            context_options = []
            for pref in preferences:
                context_options += [{'action':pref['action'], 'explanations':pref["explanations"], 'explanations_opposing':pref['explanations_opposing'], 'context_constraints':[(p,True)]} for p in pref['conditions']]
                if len(pref['conditions']) == 0:
                    context_options.append({'action':pref['action'], 'explanations':pref["explanations"], 'explanations_opposing':pref['explanations_opposing'], 'context_constraints':[(cc[0],False) for c in context_options for cc in c['context_constraints']]})
            for context in context_options:
                contextual_action = {k:activity[k] for k in ['action','activity','object','location']}
                contextual_action['action_index'] = i_act
                contextual_action['context'] = context['context_constraints']
                contextual_action['preference'] = context["action"]
                contextual_action['explanation'] = context['explanations']
                if split == 'test':
                    contextual_action['num_precedents'] = max([num_precedents(explanation) for explanation in context['explanations']] + [0])
                contextual_action['explanation_context'] = context['context_constraints']
                contextual_action['explanation_opposing'] = context["explanations_opposing"]
                open(expanded_preference_file[split], 'a').write(json.dumps(contextual_action)+'\n')

    for split in ['train','test']:
        for space in ['action','activity','object','location']:
            open(concept_file[split][space],'w').write('')
            for explanation in list_of_explanations[split][space]:
                open(concept_file[split][space],'a').write(json.dumps({'item':explanation[0],'concept':explanation[1],'match':10 if explanation[2] else 0, 'action_idx':explanation[3], "source":"user"})+'\n')

    for space in ['action','activity','object','location']:
        concepts_generated = f'data/all_concept_map_{space}_gpt4.jsonl'
        split_files = {split:os.path.join(datadir, fold_str, f'{split}_concept_map_{space}.jsonl') for split in ['train','test']}
        open(split_files['train'],'w').write('')
        open(split_files['test'],'w').write('')
        concepts_in_persona = concepts['input'][space]
        for line in open(concepts_generated).readlines():
            datapoint = json.loads(line)
            if datapoint['concept'] not in concepts_in_persona:
                continue
            if datapoint['item'] in all_sets['train'][space]:
                datapoint['action_idx'] = min(all_sets['train'][space][datapoint['item']])
                open(split_files['train'],'a').write(json.dumps(datapoint)+'\n')
            elif datapoint['item'] in all_sets['test'][space]:
                datapoint['action_idx'] = min(all_sets['test'][space][datapoint['item']])
                open(split_files['test'],'a').write(json.dumps(datapoint)+'\n')
