import os
import csv
import json
import random

persona_list = ['personaA','personaB','personaC','personaD','personaE']

for persona in persona_list:
    
    start = input(f"Press enter to continue processing persona {persona}...")
    
    raw_datafile = f'data/raw_data/raw_{persona}.csv'

    os.makedirs(f'data/{persona}', exist_ok=True)

    given_actions_file = 'data/actions_given.jsonl'
    previous_given_actions=[]
    if open(given_actions_file).read().strip() != '':
        previous_given_actions = [l.strip() for l in open(given_actions_file).readlines()]

    preference_file = os.path.join('data', persona, 'preferences.jsonl')
    open(preference_file, 'w').write('')


    all_preference_lines = []
    new_given_actions = []
    with open(raw_datafile, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for i_row,row in enumerate(reader):
            if i_row == 0: 
                i_action = row.index('Action')
                i_activity = row.index('Activity')
                i_object = row.index('Objects')
                i_location = row.index('Locations')
                i_pref = row.index('Preference')
                i_expl = row.index('Explanation')
                i_expl_opposing = row.index('Opposing Concepts')
                i_conditions = row.index('Conditions')
                i_pref_cond = row.index('Preference Conditional')
                i_expl_cond = row.index('Explanation Conditional')
                continue
            
            preference = row[i_pref].strip().lower()
            
            action_json = {
                'action': row[i_action].strip().lower(),
                'activity': row[i_activity].strip().lower(),
                'object': [r.strip().lower() for r in row[i_object].split(',')],
                'location': [r.strip().lower() for r in row[i_location].split(',')],
            }
            new_given_actions.append(json.dumps(action_json))
            
            def process_expl(expl_str):
                expl_list_in = [e.lower().strip() for e in expl_str.split(',')]
                if expl_list_in == ['']: 
                    import pdb; pdb.set_trace()
                    return ['']
                if len(expl_list_in) == 3:
                    assert expl_list_in[1] in action_json[expl_list_in[0]], f"Expl: {expl_list_in} not valid for Action: {action_json}"
                    return expl_list_in
                if expl_list_in[0] in action_json.keys():
                    if expl_list_in[0] in ['action', 'activity']:
                        expl_list_in = [expl_list_in[0], action_json[expl_list_in[0]], expl_list_in[1]]
                    else:
                        assert len(action_json[expl_list_in[0]]) == 1, f"Expl: {expl_list_in} not valid for Action: {action_json}. Too many elements"
                        expl_list_in = [expl_list_in[0], action_json[expl_list_in[0]][0], expl_list_in[1]]
                    return expl_list_in
                else:
                    if expl_list_in[0] == action_json['action']:
                        expl_list_in = ['action', action_json['action'], expl_list_in[1]]
                    elif expl_list_in[0] == action_json['activity']:
                        expl_list_in = ['activity', action_json['activity'], expl_list_in[1]]
                    elif expl_list_in[0] in action_json['object']:
                        expl_list_in = ['object', expl_list_in[0], expl_list_in[1]]
                    elif expl_list_in[0] in action_json['location']:
                        expl_list_in = ['location', expl_list_in[0], expl_list_in[1]]
                    else:
                        raise Exception(f"NO WAY TO MATCH {expl_list_in} for action {action_json}")
                    return expl_list_in
            
            explanations = [process_expl(expls) for expls in row[i_expl].strip().lower().split(';') if expls != '']
            explanations_opposing = [process_expl(expls) for expls in row[i_expl_opposing].strip().lower().split(';') if expls != '']
            explanations_cond = [process_expl(expls) for expls in row[i_expl_cond].strip().lower().split(';') if expls != '']
            conditions = [e.lower().strip() for e in row[i_conditions].split(',')] 
            
            if explanations == [['']]: explanations = []
            if explanations_opposing == [['']]: explanations_opposing = []
            conditions = [c for c in conditions if c != '']
            preference_cond = row[i_pref_cond].strip().lower()

            if any([e is None for e in explanations+explanations_opposing+explanations_cond]):
                import pdb; pdb.set_trace()

            preferences_combined = []
            if len(conditions) > 0:
                preferences_combined += [{'action':preference_cond, 'explanations':explanations+explanations_cond, 'explanations_opposing':explanations_opposing, 'conditions':conditions}]
            preferences_combined += [{'action':preference, 'explanations':explanations, 'explanations_opposing':explanations_opposing, 'conditions':[]}]
            
            action_json.update({
                'preferences': preferences_combined
                })
            
            all_preference_lines.append(json.dumps(action_json)+'\n')

    random.shuffle(all_preference_lines)
    for line in all_preference_lines:  
        open(preference_file, 'a').write(line)
        
    new_given_actions.sort(key = lambda x: json.loads(x)['action'])

    def same(str1, str2):
        data1 = json.loads(str1)
        data2 = json.loads(str2)
        if data1['action'] != data2['action']:
            return False
        if data1['activity'] != data2['activity']:
            return False
        data1['object'].sort()
        data2['object'].sort()
        if data1['object'] != data2['object']:
            return False
        data1['location'].sort()
        data2['location'].sort()
        if data1['location'] != data2['location']:
            return False
        return True

    actions_to_add = []
    for action in new_given_actions:
        if action not in previous_given_actions:
            if json.loads(action)['action'] in [json.loads(a)['action'] for a in previous_given_actions]:
                if not any([same(action, a) for a in previous_given_actions]):
                    print(f"Action {json.loads(action)['action']} already in given actions")
                    print(action)
                    print([a for a in previous_given_actions if json.loads(a)['action']==json.loads(action)['action']])
                    import pdb; pdb.set_trace()
                else:
                    continue
            actions_to_add.append(action)
            print(f"New action: {action}")

    previous_given_actions += actions_to_add
    previous_given_actions.sort(key = lambda x: json.loads(x)['action'])
    open(given_actions_file, 'w').write('\n'.join(previous_given_actions))


    ## Extract  concepts
    all_preferences = [json.loads(l.strip()) for l in open(preference_file).readlines()]
        
    concepts = {'input':{space:set() for space in ['action','activity','object','location']},
                'context':set()}

    for preference_item in all_preferences:
        for preference in preference_item['preferences']:
            all_explanations = preference['explanations'] + preference['explanations_opposing']
            for explanation in all_explanations:
                try:
                    concepts['input'][explanation[0]].add(explanation[2])
                except:
                    import pdb; pdb.set_trace()
            for condition in preference['conditions']:
                concepts['context'].add(condition)

    print("********************************************************************************")
    print(f"Concepts recovered from persona {persona}:")
    for space in concepts['input']:
        print(f"{space}:")
        print('\n\t-'+'\n\t-'.join(concepts['input'][space]))
        concepts['input'][space] = list(concepts['input'][space])
    print(f"Contexts:")
    print('\n\t-'+'\n\t-'.join(concepts['context']))
    concepts['context'] = list(concepts['context'])
    print("********************************************************************************")

    ## Add concepts from master list
    all_concepts = json.load(open('data/all_concepts.json'))

    for context in concepts['context']:
        if context not in all_concepts['context']:
            all_concepts['context'].append(context)
    all_concepts['context'].sort()

    json.dump(all_concepts, open(f'data/all_concepts.json', 'w'), indent=4)

    for space in all_concepts['input']:
        for concept in all_concepts['input'][space]:
            if concept not in concepts['input'][space]:
                concepts['input'][space].append(concept)
        concepts['input'][space].sort()
        
    for context in all_concepts['context']:
        if context not in concepts['context']:
            concepts['context'].append(context)

    concepts['output']= ["do_now",
                        "do_later",
                        "no_action",
                        "remind"]
    concepts['context'].sort()

    json.dump(concepts, open(f'data/{persona}/all_concepts.json', 'w'), indent=4)

    all_all_concepts = json.load(open('data/all_all_concepts.json'))

    for space in concepts['input']:
        for concept in concepts['input'][space]:
            if concept not in all_all_concepts['input'][space]:
                all_all_concepts['input'][space].append(concept)
        all_all_concepts['input'][space].sort()

    for context in concepts['context']:
        if context not in all_all_concepts['context']:
            all_all_concepts['context'].append(context)
    all_all_concepts['context'].sort()

    for concept in concepts['output']:
        if concept not in all_all_concepts['output']:
            all_all_concepts['output'].append(concept)
    all_all_concepts['output'].sort()

    json.dump(all_all_concepts, open(f'data/all_all_concepts.json', 'w'), indent=4)