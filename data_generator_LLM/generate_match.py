import os
import json
from gpt3_functions import generate_llm_response
from tqdm import tqdm

GEN = True
print(f"Using key: {os.environ['OPENAI_API_KEY']}")

if GEN:
    actions = [json.loads(l.strip()) for l in open('../data/actions_gen.jsonl').readlines()]
else:
    split='all'
    if split == 'all':
        actions = [json.loads(l.strip()) for l in open('../data/actions_given.jsonl').readlines()]
    else:
        actions = [a for a in [json.loads(l.strip()) for l in open('../data/actions_given.jsonl').readlines()] if a['action'] in json.load(open('../data/action_splits.json'))[split]]

concepts = json.load(open('../data/concepts.json'))
all_concepts = json.load(open('../data/all_all_concepts.json'))['input']
for space in all_concepts:
    for concept in all_concepts[space]:
        if concept not in concepts[space]:
            print(f"Concept not found! {concept}")
assert all([all([cc in concepts[k] for cc in all_concepts[k]]) for k in all_concepts.keys()])

if GEN:
    concepts_files = {k:f'../data/gen_concept_map_{k}.jsonl' for k in concepts}
else:
    concepts_files = {k:f'../data/{split}_concept_map_{k}.jsonl' for k in concepts}

if GEN:
    raw_logs_file = '../data/raw_logs_concept_match.txt'
else:
    raw_logs_file = f'../data/raw_logs_{split}_concept_match.txt'

ignore_spaces = []

done_concepts_by_item = {k:{} for k in concepts}
for concept_space,f in concepts_files.items():
    if not os.path.exists(f):
        open(f,'w').write('')
        continue
    done_this = set([(json.loads(l.strip())['item'], json.loads(l.strip())['concept']) for l in open(f).readlines()])
    done_concepts_by_item[concept_space] = {item:[] for item in set([elem[0] for elem in done_this])}
    for item,concept in done_this:
        done_concepts_by_item[concept_space][item].append(concept)
    print(f"Already done {len(done_this)} datapoints for {len(done_concepts_by_item[concept_space])} items in {concept_space} space")


for action_breakdown in tqdm(actions):
    for space, concept_list in concepts.items():
        if space in ignore_spaces: continue
        space_items = action_breakdown[space]
        if isinstance(space_items, str): space_items = [space_items]
        for space_item in space_items:
            space_item = space_item.lower()
            if space_item not in done_concepts_by_item[space]: done_concepts_by_item[space][space_item] = []
            for concept, examples in concept_list.items():
                if concept in done_concepts_by_item[space][space_item]: continue
                concept_match = {'item':space_item,'concept':concept}
                ex_pos = examples[0].lower()
                ex_neg = examples[1].lower()
                prompt_phrase = [{"role": "system", "content": "You are a helpful assistant. You will answer every question with a single number on a scale of 1-10."},
                                 {"role": "user", "content": f"How true is the statement: {ex_pos} is a/an {space} that {concept}?"},
                                 {"role": "assistant", "content": "10"},
                                 {"role": "user", "content": f"How true is the statement: {ex_neg} is a/an {space} that {concept}?"},
                                 {"role": "assistant", "content": "1"},
                                 {"role": "user", "content": f"How true is the statement: {space_item} is a/an {space} that {concept}?"}]

                result_llm = generate_llm_response(prompt_phrase).strip()

                for msg in prompt_phrase:
                    open(raw_logs_file, 'a').write("\n"+f"{msg['role'].capitalize()}: {msg['content']}\n")
                open(raw_logs_file, 'a').write("\n"+f"> Assistant: {result_llm}")
                open(raw_logs_file, 'a').write("\n"+"\n==================================\n")
                
                try:
                    concept_match['match'] = int(result_llm)
                    open(concepts_files[space],'a').write(json.dumps(concept_match)+'\n')
                except:
                    open(raw_logs_file, 'a').write(f"Error: Not convertible to integer: {result_llm}")
                    continue
                done_concepts_by_item[space][space_item].append(concept)
