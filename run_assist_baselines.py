import yaml
import json
import os
import shutil
import argparse
import time
os.environ['TOKENIZERS_PARALLELISM']='false'

import torch
from copy import deepcopy
from utils.expand_test_data import expand_test_data
from dataloader.loader_assistanceAction import AssistanceActionDataset
from baselines_assistance.rules import BaselineAssistance_RuleBased
from baselines_assistance.gpt import BaselineAssistance_GPT

torch.autograd.set_detect_anomaly(True)

RUN_GPT_BASELINE = True
RUN_RULE_BASELINE = True

DISABLE_WANDB = False
wandb_mode = "online" if not DISABLE_WANDB else "disabled"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='??????')
    
    parser.add_argument('--datapath', type=str, default='data', help='Path where the data lives. Must contain routines, info and classes json files.')
    parser.add_argument('--logs_dir', default='logs/logs', type=str, help='Path to store outputs')
    parser.add_argument('--logs_prefix', default='', type=str, help='Prefix to persona files')
    parser.add_argument('--no_concept_finetuning', type=str, default='False')
    parser.add_argument('--eval_previous_all', type=str, default='False')
    parser.add_argument('--eval_previous_concepts', type=str, default='False')
    parser.add_argument('--eval_previous_assist', type=str, default='False')
    parser.add_argument('--persona', type=str, required=True)
    
    # Optional Arguments
    parser.add_argument('--epochs_representation', type=int, default=2000)
    parser.add_argument('--lr_pretrain', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--epochs_representation_finetune', type=int, default=1000)
    parser.add_argument('--gen_concept_weight', type=float, default=0.01)
    parser.add_argument('--user_concept_weight', type=float, default=1.0)
    parser.add_argument('--given_action_concept_weight', type=float, default=0.5)    
    parser.add_argument('--batch_size_assist', type=int, default=1)
    parser.add_argument('--batch_size_concept', type=int, default=256)
    parser.add_argument('--val_fraction_assistance', type=float, default=0.1)
    parser.add_argument('--val_fraction_concepts', type=float, default=0.05)
    parser.add_argument('--test_fraction', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs_assistance', type=int, default=1700)
    parser.add_argument('--embedder', type=str, default='sentence')
    parser.add_argument('--concept_models_LR_factor', type=float, default=0.05)
    parser.add_argument('--hidden_concept_dim', type=int, default=256)
    parser.add_argument('--latent_map_dim', type=int, default=64)
    parser.add_argument('--latent_map_dim_assistance', type=int, default=96)
    parser.add_argument('--latent_map_dim_assistance_magnitude', type=int, default=32)
    parser.add_argument('--latent_map_dim_assistance_concept', type=int, default=32)
    parser.add_argument('--latent_map_dim_assistance_role', type=int, default=32)
    parser.add_argument('--concept_emb_for_assistance', type=str, default='bert')
    parser.add_argument('--loss_explanation_weight', type=float, default=20.0)
    parser.add_argument('--expl_train_attn', type=str, default='True')
    parser.add_argument('--continue_training', type=str, default='True')
    parser.add_argument('--num_train_actions', type=int, default=-1)
    parser.add_argument('--label_min', type=int, default=1)
    parser.add_argument('--label_max', type=int, default=10)
    
    cfg = parser.parse_args()
    cfg.persona = [p.strip() for p in cfg.persona.split(',')]
    cfg.eval_previous_all = cfg.eval_previous_all.lower() == 'true'
    cfg.eval_previous_concepts = cfg.eval_previous_concepts.lower() == 'true'
    cfg.eval_previous_assist = cfg.eval_previous_assist.lower() == 'true'
    cfg.expl_train_attn = cfg.expl_train_attn.lower() == 'true'
    cfg.no_concept_finetuning = cfg.no_concept_finetuning.lower() == 'true'
    cfg.continue_training = cfg.continue_training.lower() == 'true'
    
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)

    fold_nums = 3
    
    cfg.logs_dir = os.path.join(cfg.logs_dir, 'baseline_assist')
    os.makedirs(cfg.logs_dir, exist_ok=True)
    print(f"Logging to {cfg.logs_dir}")
    original_logs_dir = cfg.logs_dir

    embedding_path = {k: os.path.join(cfg.datapath, f'gen_concept_map_{k}.jsonl').replace('concept_','embedding_').replace('.jsonl','.pt').replace('.json','.pt') for k in ['action','activity','location','object']}
    
    personas = deepcopy(cfg.persona)
    
    for persona in personas:
        all_concepts = json.load(open(os.path.join(cfg.datapath, persona, 'all_concepts.json')))
        concept_list_overall = []
        for concept_space in ['action','activity','location','object']:
            concept_list_overall += list(all_concepts['input'][concept_space])
        cfg.num_concepts_overall = len(concept_list_overall)
        category_list = []
        for fold_num in range(fold_nums):
            expand_test_data(persona, fold_num = fold_num, num_folds = fold_nums)
            persona_datadir = os.path.join(cfg.datapath, persona, 'fold_{:02d}'.format(fold_num))
            if RUN_RULE_BASELINE:
                baseline_assistance_rule = BaselineAssistance_RuleBased(persona_datadir, 
                                                                        persona, 
                                                                        logs_dir=os.path.join(original_logs_dir, persona, 'rules', 'Seen_{:02d}'.format(cfg.num_train_actions)),
                                                                        num_train_actions=cfg.num_train_actions)
            if RUN_GPT_BASELINE:
                baseline_assistance_gpt = BaselineAssistance_GPT(persona_datadir, 
                                                                 persona, 
                                                                 logs_dir=os.path.join(original_logs_dir, persona, 'gpt', 'Seen_{:02d}'.format(cfg.num_train_actions)),
                                                                 num_train_actions=cfg.num_train_actions)
        
            print(f"\n\nMoving on to assistance preferences for {persona}-{fold_num}\n\n")
            
            cfg.persona = persona
            data_assist = AssistanceActionDataset(dirpath=persona_datadir, concept_src='gpt', **cfg.__dict__)
            if not data_assist.has_data():
                print(f"\n\nWARNING: Assistance data from user is empty!\n\n")
                continue

            cfg.action_vocab = data_assist.action_vocab

            if data_assist.has_data(type_data='test'):
                if RUN_RULE_BASELINE:
                    print("Running Rule Baseline...................")
                    baseline_assistance_rule.run(files_prefix='{:02d}'.format(fold_num))
                    pass
                if RUN_GPT_BASELINE:
                    print("Running GPT Baseline...................")
                    baseline_assistance_gpt.run(files_prefix='{:02d}'.format(fold_num))
                    pass
            else:
                print(f"\n\nWARNING: Assistance test data from user for fold is empty!\n\n")

