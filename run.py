import yaml
import json
import os
import shutil
import argparse
import time
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['CUDA_LAUNCH_BLOCKING']='1'

import numpy as np
from math import ceil
import torch
import wandb
from copy import deepcopy
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import multiprocessing as mp

torch.set_float32_matmul_precision('high')

from utils.expand_test_data import expand_test_data
from utils.visualization_helpers import results_visualizer_assistance, visualize_combined_results, visualize_online
from utils.eval_helpers import combine_confusion_matrix_dicts, ConfusionMatrix
from dataloader.loader_assistanceAction import AssistanceActionDataset
from model.AssistancePredictionModule import AssistancePredictionModule
from model.AssistancePredictionDirectModule import AssistancePredictionDirectModule

torch.autograd.set_detect_anomaly(True)

enable_progress_bar=True
MULTIPROC = False
DISABLE_WANDB = True
CONCEPT_SRCS = ['no_concepts','gpt','user']
MODEL_CHECK = True
wandb_mode = "online" if not DISABLE_WANDB else "disabled"

def compare_models(model_1, model_2, num_params_expected=None):
    model_2.to(model_1.device)
    models_differ = 0
    num_params = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            num_params += torch.numel(key_item_1[1])
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('!!! Mismatch found at', key_item_1[0])
            else:
                raise Exception
    assert num_params_expected is None or num_params_expected == num_params, f"Expected {num_params_expected} got {num_params}"
    if models_differ == 0:
        print(f'\nModels match perfectly! :) All {num_params} parameters!!!\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='??????')
    
    parser.add_argument('--datapath', type=str, default='data', help='Path where the data lives. Must contain routines, info and classes json files.')
    parser.add_argument('--logs_dir', default='logs/logs', type=str, help='Path to store outputs')
    parser.add_argument('--logs_prefix', default='', type=str, help='Prefix to persona files')
    parser.add_argument('--eval_previous', type=str, default='False')
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
    cfg.eval_previous = cfg.eval_previous.lower() == 'true'
    cfg.expl_train_attn = cfg.expl_train_attn.lower() == 'true'
    cfg.continue_training = cfg.continue_training.lower() == 'true'
    
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)

    for key,value in cfg.__dict__.items():
        print(f"\t\t- {key}: {value}")
    
    personas = deepcopy(cfg.persona)
    
    os.makedirs(cfg.logs_dir, exist_ok=True)

    embedding_path = {k: os.path.join(cfg.datapath, f'gen_embedding_map_{k}.pt') for k in ['action','activity','location','object']}
    
    
    original_logs_dir = cfg.logs_dir
    train_size_logs_dir = os.path.join(cfg.logs_dir, 'Seen_{:02d}'.format(cfg.num_train_actions))
    
    combined_metrics = {'personas':personas,'left_to_do':[]}
    
    for persona_global in personas:
        all_concepts = json.load(open(os.path.join(cfg.datapath, persona_global, 'all_concepts.json')))
        concept_spaces_used = [k for k in ['action','activity','location','object'] if len(all_concepts['input'][k]) > 0]
        print(f"Persona {persona_global} used concept spaces: {concept_spaces_used}")
        concept_list_overall = []
        for concept_space in ['action','activity','location','object']:
            concept_list_overall += list(all_concepts['input'][concept_space])
        cfg.num_concepts_overall = len(concept_list_overall)
        viz_results = {'object':{'item_embeddings':torch.tensor([]), 'concept_embeddings':torch.tensor([]), 'item_categories':[], 'concepts_categories':[]},
                    'location':{'item_embeddings':torch.tensor([]), 'concept_embeddings':torch.tensor([]), 'item_categories':[], 'concepts_categories':[]},
                    'activity':{'item_embeddings':torch.tensor([]), 'concept_embeddings':torch.tensor([]), 'item_categories':[], 'concepts_categories':[]},
                    'action':{'item_embeddings':torch.tensor([]), 'concept_embeddings':torch.tensor([]), 'item_categories':[], 'concepts_categories':[]}}
        
        category_list = []
        fold_nums = 3

        cfg.logs_dir = original_logs_dir
        def train_assistance(fold_num, persona_local, concept_src):
            output_dir = os.path.join(train_size_logs_dir, persona_local, 'Fold_{:02d}'.format(fold_num))
            cfg.persona = persona_local
            cfg.concept_src = concept_src
            
            persona_datadir = os.path.join(cfg.datapath, persona_local, 'fold_{:02d}'.format(fold_num))
            expand_test_data(persona_local, fold_num = fold_num, num_folds=fold_nums)
            
            if concept_src != 'model' and cfg.concept_emb_for_assistance == 'latent': return
            output_dir_assist = os.path.join(output_dir, cfg.logs_prefix+f'{concept_src}_concepts')
            if cfg.concept_emb_for_assistance == 'latent':
                output_dir_assist += '_latent'
            if os.path.exists(output_dir_assist): 
                print(f"Assistance model exists at {output_dir_assist}.", end = ' ')
                if not cfg.eval_previous:
                    print("Skipping...")
                    return
                print("Loading...")
                assert 'Fold_{:02d}'.format(fold_num) in output_dir, f"Fold number {fold_num} not found in logs_dir: {output_dir}"
                if concept_src == 'no_concepts':
                    model_assistance = AssistancePredictionDirectModule.load_from(output_dir_assist, all_concepts=json.load(open(os.path.join(persona_datadir, 'all_concepts.json'))), config=cfg.__dict__)
                else:
                    model_assistance = AssistancePredictionModule.load_from(output_dir_assist, all_concepts=json.load(open(os.path.join(persona_datadir, 'all_concepts.json'))), config=cfg.__dict__)
                epochs = 0
            
            else:
                os.makedirs(output_dir_assist, exist_ok=True)
                if concept_src == 'no_concepts':
                    model_assistance = AssistancePredictionDirectModule(all_concepts=json.load(open(os.path.join(persona_datadir, 'all_concepts.json'))), **cfg.__dict__)
                else:
                    model_assistance = AssistancePredictionModule(all_concepts=json.load(open(os.path.join(persona_datadir, 'all_concepts.json'))), **cfg.__dict__)
                epochs = cfg.epochs_assistance
                print(f"\n\nTraining assistance for {persona_local} at {output_dir_assist}")

            wandb_logger = WandbLogger(project="adaptive-personalization-assistance", settings=wandb.Settings(start_method="fork"), mode=wandb_mode, tags=[concept_src, cfg.logs_prefix.replace('_',' ').strip(), persona_local])
            wandb_logger.experiment.config.update(cfg, allow_val_change=True)
        
            data_assist = AssistanceActionDataset(dirpath=persona_datadir, no_validation=True, new_concept_files_dir=os.path.join(train_size_logs_dir, persona_local, 'Fold_{:02d}'.format(fold_num)), **cfg.__dict__)
            if not data_assist.has_data():
                print(f"\n\nWARNING: Assistance data from user is empty!")
                return

            cfg.action_vocab = data_assist.action_vocab
            ckpt_callback = ModelCheckpoint(dirpath=output_dir_assist, monitor=f"Val_ES_accuracy_assistance", filename="{epoch}-{Val_ES_accuracy_assistance}", mode="max", save_top_k=1000, save_last=True, every_n_epochs=20, auto_insert_metric_name=True)
            trainer = Trainer(accelerator='gpu', 
                                devices = torch.cuda.device_count(), 
                                logger=wandb_logger, 
                                max_epochs=epochs, 
                                log_every_n_steps=1, 
                                callbacks=[ckpt_callback], 
                                check_val_every_n_epoch=1,
                                enable_progress_bar=enable_progress_bar)
            if data_assist.has_data(type_data='train'):
                if data_assist.has_data(type_data='test'):
                    trainer.fit(model_assistance, data_assist.get_train_loader(), data_assist.get_test_loader())
            else:
                print(f"\n\n!!!!!!!!!!!!!!!!!WARNING: Assistance train data from user for fold is empty!")
            
            model_assistance.save_to(output_dir_assist)
            if concept_src == 'no_concepts':
                model_loaded = AssistancePredictionDirectModule.load_from(output_dir_assist, config=cfg.__dict__).to(model_assistance.device)
            else:
                model_loaded = AssistancePredictionModule.load_from(output_dir_assist, config=cfg.__dict__).to(model_assistance.device)

            if data_assist.has_data(type_data='test'):
                model_loaded.explanations = {}
                model_loaded.explanation_gradients = {}
                model_loaded.explanation_attn_idxs = {}
                trainer.test(model_loaded, data_assist.get_test_loader())
                model_loaded.to('cuda')
                model_loaded.eval()
                for batch_idx, batch in enumerate(data_assist.get_test_loader()):
                    model_loaded.eval_with_expl(batch, batch_idx=batch_idx, action_vocab=data_assist.action_vocab)
                cm_obj = model_loaded.write_results(output_dir_assist)
            else:
                print(f"\n\nWARNING: Assistance test data from user for fold is empty!")
            print(f"\n\nTesting done for {persona_local} at {output_dir_assist}")
            open(os.path.join(train_size_logs_dir, persona_local, 'done.txt'),'a').write("A - Fold{:02d}_{}\n".format(fold_num, concept_src))
            wandb.finish()

        if MULTIPROC:
            def left_to_do(fold_num, concept_src):
                if concept_src != 'model' and cfg.concept_emb_for_assistance == 'latent': return False
                output_dir_assist = os.path.join(train_size_logs_dir, persona_global, 'Fold_{:02d}'.format(fold_num), cfg.logs_prefix+f'{concept_src}_concepts')
                if cfg.concept_emb_for_assistance == 'latent':
                    output_dir_assist += '_latent'
                if not os.path.exists(output_dir_assist): return True
                if cfg.eval_previous: return True
                if cfg.continue_training and not os.path.exists(os.path.join(output_dir_assist, 'weights.pt')): 
                    shutil.rmtree(output_dir_assist)
                    return True
                return False
            processes = [mp.Process(target=train_assistance, args=(fold_num, persona_global, concept_src)) for fold_num in range(fold_nums) for concept_src in CONCEPT_SRCS if left_to_do(fold_num, concept_src)]
            
            while True:
                if len(processes) == 0:
                    break
                # Run processes
                for i,p in enumerate(processes[:15]):
                    print(f"################################################ Starting process {i+1} ################################################")
                    p.start()
                    time.sleep(10)
                # Exit the completed processes
                for p in processes:
                    p.join()
                processes = processes[15:]

        else:
            for fold_num in range(fold_nums):
                print(f"\n\nAssistive preferences for {persona_global}-{fold_num}\n\n")
                for concept_src in CONCEPT_SRCS:
                    train_assistance(fold_num, persona_global, concept_src)
                    
        keys = {
                'model_concepts':'with_model', 
                'gpt_concepts':'with_gpt', 
                'user_concepts':'with_user', 
                'model_concepts_latent':'with_model_latent',
                'no_concepts_concepts':'with_no_concepts',
                }
        results = {'left_to_do':[]}
        results.update({k:{'cf':[], 'MRR':{}} for k in keys.values()})
        for fold_num in range(fold_nums):
            for src in keys:
                filename = os.path.join(train_size_logs_dir, persona_global, 'Fold_{:02d}'.format(fold_num), cfg.logs_prefix+src, 'assistances_humanread.json')
                if os.path.exists(filename):
                    data = json.load(open(filename))
                    results[keys[src]]['cf'].append(data['confusion_matrix_obj']['confusion_matrix'])
                    results[keys[src]]['MRR'] = {}
                    for mrr_key in data['MRR'].keys():
                        if mrr_key not in results[keys[src]]['MRR']:
                            results[keys[src]]['MRR'][mrr_key] = []
                        results[keys[src]]['MRR'][mrr_key].append(data['MRR'][mrr_key])
                else:
                    if src in ['gpt_concepts','user_concepts']:
                        open('logs/gpt4_fewCon_byEpoch/left.txt','a').write(filename+"\n")
                    results['left_to_do'].append(filename)
        for src in keys.values():
            results[src]['cf'] = combine_confusion_matrix_dicts(results[src]['cf']) if len(results[src]['cf']) > 0 else ConfusionMatrix(['do_now', 'do_later', 'remind', 'no_action']).get_numbers_dict()
            for mrr_key in results[src]['MRR'].keys():
                if len(results[src]['MRR'][mrr_key]) == 0 or sum([r[1] for r in results[src]['MRR'][mrr_key]]) == 0:
                    results[src]['MRR'][mrr_key] = 0
                else:
                    results[src]['MRR'][mrr_key] = np.average([r[0] for r in results[src]['MRR'][mrr_key]], weights=[r[1] for r in results[src]['MRR'][mrr_key]])
        json.dump(results, open(os.path.join(train_size_logs_dir, persona_global, cfg.logs_prefix+'assistances_final.json'),'w'), indent=4)
        new_entry_comb_res = results_visualizer_assistance(original_logs_dir, persona_global, num_train_actions=cfg.num_train_actions, prefix=cfg.logs_prefix)
        for key,value in new_entry_comb_res.items():
            if key not in combined_metrics:
                combined_metrics[key] = {}
            for k,v in value.items():
                if k not in combined_metrics[key]:
                    combined_metrics[key][k] = []
                combined_metrics[key][k].append(v)
        combined_metrics['left_to_do'] += results['left_to_do']
    combined_metrics['left_to_do'].sort()
    visualize_combined_results(combined_metrics, train_size_logs_dir, logs_prefix=cfg.logs_prefix+'_{:02d}'.format(cfg.num_train_actions))
    visualize_online(original_logs_dir, cfg.logs_prefix)
