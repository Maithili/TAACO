import os
import random
import json
from tqdm import tqdm

from baselines_assistance.base import BaselineAssistance
from data_generator_LLM.gpt3_functions import generate_llm_response, convert_numbered_list_to_list
from utils.eval_helpers import complete_evals

VERBOSE = True

def phrase_action(action):
    action_str = action['action'][0] if isinstance(action['action'], list) else action['action']
    ativity_str = action['activity'][0] if isinstance(action['activity'], list) else action['activity']
    phrase = f"the action {action_str}, towards {ativity_str}, using {', '.join(action['object'])}, at {', '.join(action['location'])}"
    return phrase

def phrase_explanation(explanation_given):
        article = 'an' if explanation_given[0][0].lower() in ['a','e','i','o','u'] else 'a'
        expl = f"{explanation_given[1]} is {article} {explanation_given[0]} which is {explanation_given[2]}"
        return expl
    
class PreferencePrompt():
    def __init__(self, concepts_file, default_action=None) -> None:
        self.prompt = ""
        self.concepts = json.load(open(concepts_file))['input']
        self.default_action = default_action
        

    def add_preference_datapoint(self, datapoint):
        preference_list = datapoint['preferences']
        if len(preference_list) == 0:
            raise ValueError("Preference list is empty")
        action = preference_list[0]['action'].replace(' ','_')
        context = ''
        if len(preference_list[0]['conditions']) > 0:
            context = f" when {', '.join(preference_list[0]['conditions'])}"
        explanation = ''
        if len(preference_list[0]['explanations']) > 0:        
            explanation = ', because '+ ', and '.join([f"{d[1]} is a/an {d[0]} which is {d[2]}" for d in preference_list[0]['explanations']])
        context_else = ''
        explanation_else = ''
        if len(preference_list) > 1:
            assert len(preference_list) == 2, f"More than 2 preferences not supported; got {len(preference_list)}"
            action2 = preference_list[1]['action'].replace(' ','_')
            context_else += f", otherwise {action2}"
            explanation_else = ''
            if len(preference_list[1]['explanations']) > 0:
                explanation_else = ', because '+ ', and '.join([f"{d[1]} is a/an {d[0]} which is {d[2]}" for d in preference_list[1]['explanations']])
        phrase = f"The user wants the robot's assistance with the action {phrase_action(datapoint)} to be {action}{context}{explanation}{context_else}{explanation_else}."
        self.prompt += phrase + "\n"
    
    def eval_explanation(self, explanations_given, explanations_expected, raw_logs_file=None):
        assert len(explanations_given) <=1, f"More than 1 explanation not supported; got {len(explanations_given)}"
        if len(explanations_given) == 0:
            return 0
        explanation_given = explanations_given[0]
        if explanation_given == '': return 0
        system_prompt = "You are a helpful assistant. You will help evaluate whether the given explanations have the same meaning. You will answer with a single word out of 'yes' or 'no'."
        for explanation_expected in explanations_expected:
            both_explanations = [explanation_given, phrase_explanation(explanation_expected)]
            random.shuffle(both_explanations)
            ask_prompt = f"Are the following explanations equivalent?\n\n{both_explanations[0]}\n\n{both_explanations[1]}"
            full_prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ask_prompt},
            ]
            result_llm = generate_llm_response(full_prompt).replace('.','').replace(',','').strip()
            if raw_logs_file is not None:
                open(raw_logs_file, 'a').write("\n"+result_llm+"<--")
                open(raw_logs_file, 'a').write("\n"+ask_prompt)
            if result_llm.lower() == 'yes':
                return 1
            else:
                if result_llm.lower() != 'no':
                    print(f"WARNING: GPT-3 returned an unexpected evaluation: {result_llm}")
                    import pdb; pdb.set_trace()
        return 0
        
    
    def get_preference(self, datapoint):
        system_prompt = "You are a helpful assistant. You will help a robot decide how to assist their user with a given activity, based on the user's preferences. You will answer with a single phrase out of 'no_action', 'do_now', 'do_later', 'remind'. You will NOT explain your answer."
        context = f", when {','.join(datapoint['context'])}"
        ask_prompt = f"How would the user want the robot's assistance with the action {phrase_action(datapoint)}{context} to be?"
        full_prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.prompt + ask_prompt},
        ]
        result_llm = generate_llm_response(full_prompt).strip()
        
        if 'because' in result_llm:
            prediction = result_llm[:result_llm.index('because')]
            explanations = result_llm[result_llm.index('because'):]
        elif '\n' in result_llm:
            prediction = result_llm[:result_llm.index('\n')]
            explanations = result_llm[result_llm.index('\n')+1:]
        else:
            prediction = result_llm
            explanations = ''
        
        if prediction not in ['no_action', 'do_now', 'do_later', 'remind']:
            if prediction.startswith('do now'): 
                prediction = 'do_now'
            elif prediction.startswith('do later'): 
                prediction = 'do_later'
            elif prediction.startswith('remind me'): 
                prediction = 'remind'
            elif prediction.startswith('no action'): 
                prediction = 'no_action'
            else:
                print(f"WARNING: GPT-3 returned an unexpected action: {prediction}")
                return None, None

        ## Explain!
        system_prompt2 = "You are a helpful assistant. You will help a robot decide how to assist their user with a given activity, based on the user's preferences. "
        system_explanation = "You will explain your reasoning through a numbered list of explanations, similar to the ones provided in the examples. Each explanation must follow the template: '<item> is a/an <category> which is <property>'.\n"
        full_prompt = [
            {"role": "system", "content": system_prompt2},
            {"role": "user", "content": self.prompt + ask_prompt},
            {"role": "assistant", "content": prediction},
            {"role": "system", "content": system_explanation},
            {"role": "user", "content": "Why did you choose this preference?"}
        ]
        explanations = generate_llm_response(full_prompt).strip()

        return prediction.replace(' ','_'), explanations

class BaselineAssistance_GPT(BaselineAssistance):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, files_prefix):
        output_file = os.path.join(self.logs_dir, f'{files_prefix}output.json')
        if os.path.exists(output_file):
            print(f"Output exists at {output_file}. Skipping...")
            return
        raw_logs_file = os.path.join(self.logs_dir, f'{files_prefix}raw_logs.txt')
        open(raw_logs_file,'w').write('')
        preferences_train, preferences_test = self.get_train_test_preferences()

        gpt_based = PreferencePrompt(concepts_file=os.path.join(self.datadir,'all_concepts.json'))
        for p in preferences_train:
            gpt_based.add_preference_datapoint(p)

        open(raw_logs_file, 'a').write("\n"+"PROMPT:")
        open(raw_logs_file, 'a').write(gpt_based.prompt)
        open(raw_logs_file, 'a').write("\n\n\n")

        result = {'accuracy':None,'num_episodes':None,'confusion_matrix':None, 'confusion_matrix_norm':None,'episodes':[]}

        episodes = {}
        print(f"Writing to {raw_logs_file}, {output_file}")
        for p in tqdm(preferences_test):
            if 'preference' not in p or p['preference'] is None: continue
            expectation = p['preference'].replace(' ','_') if 'preference' in p else None
            prediction, explanation = gpt_based.get_preference(p)
            if explanation is not None:
                all_explanations = convert_numbered_list_to_list(explanation)
                first_explanation = all_explanations[0] if len(all_explanations) > 0 else ''
            else:
                first_explanation = ''
            if prediction is None: 
                prediction = 'do_now'
            episodes[self._key_from_text(p)] = {
                'ground_truth': expectation,
                'prediction': prediction,
                'result': 'correct' if prediction == expectation else 'wrong',
                'explanation': p['explanation'],
                'explanation_gpt': explanation,
                'mrr': gpt_based.eval_explanation([first_explanation], p['explanation'], raw_logs_file=raw_logs_file),
            }
            open(raw_logs_file, 'a').write("\n"+f"{prediction} (GT){expectation} <-- {str(p)}\n\n")
            
        result = complete_evals([(r['prediction'], r['ground_truth']) for r in episodes.values()], do_now_in_outputs=True)
        mask_list = [r for r in episodes.values() if len(r['explanation'])>0]
        correct_mask = [r['result']=='correct' for r in mask_list]
        masked_series = [r['mrr'] for r in mask_list]
        result["MRR"] = sum(masked_series)/len(masked_series), len(masked_series)
        masked_series = [r['mrr'] for r,f in zip(mask_list,correct_mask) if f]
        result["MRR_correct"] = sum(masked_series)/len(masked_series), len(masked_series)
        masked_series = [r['mrr'] for r,f in zip(mask_list,correct_mask) if not f]
        result["MRR_wrong"] = sum(masked_series)/len(masked_series), len(masked_series)
        
        result['episodes'] = episodes

        self.update_confusion_matrix(result['confusion_matrix_obj'], {'overall':result["MRR"], 'correct':result['MRR_correct'], 'wrong':result['MRR_wrong']})
        result['confusion_matrix_obj'] = result['confusion_matrix_obj'].get_numbers_dict()
        open(output_file,'w').write(json.dumps(result, indent=4))
        print(f"Results written to {output_file}")
