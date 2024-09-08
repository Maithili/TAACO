import os
import json
import time
import random
import sys
sys.path.append('model')
from copy import deepcopy
from typing import Optional
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from pytorch_lightning import LightningModule
from eval_helpers import complete_evals
from embedders import get_embedder

class AssistancePredictionModule(LightningModule):
    '''
    MLP plus backprop for explanation
    '''
    def __init__(self, all_concepts, **kwargs):
        super().__init__()
        self.cfg = kwargs
        self.cfg['num_assistive_actions'] = len(all_concepts['output'])

        ## Constructing Role embeddings
        next_emb = 0
        self.roles = {}

        for input_concept in all_concepts['input'].keys():
            self.roles[input_concept] = torch.tensor(next_emb).clone().to('cuda')
            next_emb += 1

        self.roles['context'] = torch.tensor(next_emb).clone().to('cuda')
        self.roles['output'] = torch.tensor(next_emb).clone().to('cuda')
        next_emb += 1

        self.role_embeddings = torch.nn.Embedding(next_emb, self.cfg['latent_map_dim_assistance_role'])
        self.context_embeddings = torch.nn.Embedding(len(all_concepts['context']), self.cfg['latent_map_dim_assistance_concept'])
        self.context_embeddings_ordered = lambda : torch.stack([self.context_embeddings(torch.tensor(i).to('cuda')) for i in range(len(all_concepts['context']))], dim=0)
        
        self.concepts_emb = {}
        concept_spaces_used = [k for k in ['action','activity','location','object'] if len(all_concepts['input'][k]) > 0]
       
        assert self.cfg['concept_emb_for_assistance'] == 'bert', "Only BERT embeddings are supported for now"
        for space in concept_spaces_used:
            embedding_path = 'data/gen_embedding_map_action.pt'
            if os.path.exists(embedding_path): 
                for trial in range(10):
                    try:
                        embedding_model = get_embedder(self.cfg['embedder'], path=embedding_path)
                        break
                    except:
                        print(f"Failed to load embedding model at {embedding_path}, retrying...")
                        time.sleep(random.random()*2)
                    if trial == 9:
                        embedding_model = get_embedder(self.cfg['embedder'])
                        break
            else: 
                embedding_model = get_embedder(self.cfg['embedder'])
            embedding_model.add_concepts(all_concepts['input'][space])
            self.concepts_emb[space] = torch.cat([embedding_model.map['concepts'][c] for c in all_concepts['input'][space]], dim=0)
        
        self.cfg['concept_emb_input_dim'] = self.concepts_emb['action'][0].shape[-1]

        self.concept_projector = nn.Linear(self.cfg['concept_emb_input_dim'], self.cfg['latent_map_dim_assistance_concept'])

        ## Magnitude Projector
        self.magnitude_projector = nn.Sequential(nn.Linear(1, int(round(0.5*self.cfg['latent_map_dim_assistance_magnitude']))),
                                        nn.ReLU(),
                                        nn.Linear(int(round(0.5*self.cfg['latent_map_dim_assistance_magnitude'])), self.cfg['latent_map_dim_assistance_magnitude'])
                                        )

        ## Prediction
        self.predictor_layer0 = torch.nn.TransformerEncoderLayer(self.cfg['latent_map_dim_assistance'], nhead=2, batch_first=True, dropout=0.3)
        self.predictor_layer = torch.nn.TransformerEncoderLayer(self.cfg['latent_map_dim_assistance'], nhead=2, batch_first=True, dropout=0.3)
        self.action_predictor = nn.Sequential(nn.Linear(self.cfg['latent_map_dim_assistance'], self.cfg['latent_map_dim_assistance']),
                                        nn.ReLU(),
                                        nn.Linear(self.cfg['latent_map_dim_assistance'], self.cfg['num_assistive_actions'])
                                        )

        self.loss = nn.CrossEntropyLoss()
        self.expl_att_loss = nn.BCELoss(reduce=False)
        self.concept_hierarchy = all_concepts
        self.explanations = {}
        self.explanation_gradients = {}
        self.explanation_attn_idxs = {}
    
    def predictor(self, features, src_key_padding_mask, return_attention_weights=False):
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=None,
            other_name="src_mask",
            target_type=features.dtype
        )
        src_mask = F._canonical_mask(
            mask=None,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=features.dtype,
            check_other=False,
        )
        # self attention block
        def _sa_block(x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
            x = self.predictor_layer.self_attn(x, x, x,
                                            attn_mask=attn_mask,
                                            key_padding_mask=key_padding_mask,
                                            need_weights=return_attention_weights, is_causal=is_causal)
            return (self.predictor_layer.dropout1(x[0]), x[1])

        # feed forward block
        def _ff_block(x: Tensor) -> Tensor:
            x = self.predictor_layer.linear2(self.predictor_layer.dropout(self.predictor_layer.activation(self.predictor_layer.linear1(x))))
            return self.predictor_layer.dropout2(x)
        
        x = self.predictor_layer0(features, src_key_padding_mask=src_key_padding_mask)
        x_attn, w_attn = _sa_block(x, attn_mask = src_mask, key_padding_mask = src_key_padding_mask)
        x = self.predictor_layer.norm1(x + x_attn)
        x = self.predictor_layer.norm2(x + _ff_block(x))
        
        return x, w_attn

    
    @classmethod
    def load_from(cls, dirpath, all_concepts=None, config={}, ckpt_file=None):
        if ckpt_file is None:
            ckpt_file = 'last.ckpt'
        print(f"***** Loading model from {os.path.join(dirpath, ckpt_file)} *****")
        if all_concepts is None:
            all_concepts = json.load(open(os.path.join(dirpath, 'config.json')))['concept_hierarchy']
        config_in = {}
        if os.path.exists(os.path.join(dirpath, 'config.json')):
            config_in = json.load(open(os.path.join(dirpath, 'config.json')))
            if 'config' in config_in:
                config_in.update(config_in['config'])
            remove_keys = ['eval_previous_all','eval_previous_concepts','eval_previous_assist','no_concept_finetuning']
            config_in = {k:v for k,v in config_in.items() if k not in remove_keys}
            ## Compare whether the config is the same
            for k,v in config.items():
                if k in config_in:
                    if config_in[k] != v:
                        print(f"Config mismatch: {k}: {config_in[k]} v.s. {v}")
                else:
                    if k not in remove_keys:
                        print(f"Config mismatch: key {k} not found in the saved mcodel's config")

        model = cls.load_from_checkpoint(os.path.join(dirpath, ckpt_file), all_concepts=all_concepts, **config)
        print("***** Model Loaded *****")
        return model
 
    def save_to(self, dirpath):
        os.makedirs(dirpath, exist_ok=True)
        remove_keys = ['eval_previous_all','eval_previous_concepts','eval_previous_assist','no_concept_finetuning']
        filtered_config = {k:v for k,v in self.cfg.items() if k not in remove_keys}
        json.dump({'config':filtered_config, 'concept_hierarchy':self.concept_hierarchy}, open(os.path.join(dirpath, 'config.json'), 'w'), indent=4)
        torch.save(self.state_dict(), os.path.join(dirpath, 'weights.pt'))
        print(f"***** Model saved to {dirpath} *****")

    def _get_input_concept_similarities_from_concept_model(self, concept_model, features):
        batch, num_seq, emb_dim = features.shape
        similarities = concept_model.get_item_concept_similarities(items=features.view(batch * num_seq, emb_dim))
        similarities = similarities.view(batch, num_seq, similarities.shape[1])
        return similarities

    def construct_inputs(self, batch_inputs, introduce_input_gradients=False, from_concepts=True):
        features_encoded = []
        mask = []
        self.explanation_gradients['inputs'] = {}
        
        for key,data in batch_inputs.items():
            concepts = self.concepts_emb[key].unsqueeze(0).to(self.device)
            batch, num_seq, _ = data['features'].shape
            if from_concepts:
                assert torch.all(data['features']<=1) and torch.all(data['features']>=-1), f"Input features out of range: {data['features'].min().item()} to {data['features'].max().item()}"
                similarities = data['features']
            else:
                raise NotImplementedError("Getting input similarities within the model is out of date!")
            if introduce_input_gradients:
                similarities = similarities.detach()
                similarities = Variable(similarities, requires_grad=True)
                self.explanation_gradients['inputs'][key] = similarities
            
            num_conc = similarities.shape[2]
            assert num_conc == concepts.shape[1], f"Number of concepts in data and concept model do not match: {num_conc} v.s. {concepts.shape[1]}"

            concepts = self.concept_projector(concepts).unsqueeze(1).repeat(batch,num_seq,1,1).view(batch,num_seq*num_conc,self.cfg['latent_map_dim_assistance_concept'])
            values = self.magnitude_projector(similarities.view(batch, num_seq*num_conc, 1))
            roles = self.role_embeddings(self.roles[key]).unsqueeze(0).unsqueeze(1).repeat(batch,num_seq*num_conc,1)
            values_norm = nn.functional.normalize(values, dim=-1)
            roles_norm = nn.functional.normalize(roles, dim=-1)
            concepts_norm = nn.functional.normalize(concepts, dim=-1)
            features_encoded.append(torch.cat([values_norm, roles_norm, concepts_norm], dim=-1))
            mask.append(data['mask'].unsqueeze(-1).repeat(1,1,num_conc).view(batch,-1))
        
        idx_prev = 0
        self.explanation_attn_idxs['inputs'] = {}
        for k,feat in zip(batch_inputs.keys(), features_encoded):
            self.explanation_attn_idxs['inputs'][k] = (idx_prev, idx_prev+feat.shape[1])
            idx_prev += feat.shape[1]

        features_encoded_tensor = torch.cat(features_encoded, dim=1)
        mask_tensor = torch.cat(mask, dim=1)

        ## features: batch x input_sequence(objects+action+activity+location) x latent_map_dim
        ## mask    : batch x input_sequence(objects+action+activity+location) x 1
        return features_encoded_tensor, mask_tensor

    def construct_context(self, batch_context, introduce_input_gradients=False):
        self.explanation_gradients['context'] = {}
        self.explanation_attn_idxs['context'] = (0, batch_context['features'].shape[1])
        batch_context['features'] = batch_context['features']
        if introduce_input_gradients:
            batch_context['features'] = batch_context['features'].detach()
            batch_context['features'] = Variable(batch_context['features'], requires_grad=True)
            self.explanation_gradients['context'] = batch_context['features']
        batch_sz = batch_context['features'].shape[0]
        values = self.magnitude_projector(batch_context['features'].unsqueeze(-1))
        concepts = self.context_embeddings_ordered().unsqueeze(0).repeat(batch_sz,1,1)
        roles = self.role_embeddings(self.roles['context']).unsqueeze(0).unsqueeze(1).repeat(batch_sz,concepts.shape[1],1)
        values_norm = nn.functional.normalize(values, dim=-1)
        roles_norm = nn.functional.normalize(roles, dim=-1)
        concepts_norm = nn.functional.normalize(concepts, dim=-1)
        features_encoded_tensor = torch.cat([values_norm, roles_norm, concepts_norm], dim=-1)
        mask_tensor = torch.zeros(features_encoded_tensor.shape[0], features_encoded_tensor.shape[1]).to(features_encoded_tensor.device)
        
        ## features: batch x input_sequence(user+time) x latent_map_dim
        ## mask: batch x input_sequence(user+time) x 1
        return features_encoded_tensor, mask_tensor

    def forward(self, batch, introduce_input_gradients=False, output_as_target=False, validation=False):
        """
        Args: batch:{
            'input' : {
                'object' :   {'features' : torch.Tensor, 'mask' : torch.Tensor}, 
                'action' :   {'features' : torch.Tensor, 'mask' : torch.Tensor}, 
                'activity' : {'features' : torch.Tensor, 'mask' : torch.Tensor}, 
                'location' : {'features' : torch.Tensor, 'mask' : torch.Tensor},
            },
            'context' : {
                'user' : {
                    'state' :    {'features' : torch.Tensor, 'mask' : torch.Tensor},
                    'location' : {'features' : torch.Tensor, 'mask' : torch.Tensor},
                }
                'time' : {
                    'timeofday' : {'features' : torch.Tensor, 'mask' : torch.Tensor},
                }
            },
            'assistance': torch.Tensor
            'text':...
        }
        """

        action_logits, attention_weights, attention_mask = self.step(batch, introduce_input_gradients)

        target = batch['assistance'].to(action_logits.device).float()
        if output_as_target:
            target = F.one_hot(action_logits.argmax(-1), num_classes=self.cfg['num_assistive_actions']).clone().detach()

        loss_prediction = -(nn.LogSoftmax(dim=-1)(action_logits) * target).sum(-1).mean()
        accuracy = (target[torch.arange(target.shape[0]),action_logits.argmax(-1)]).sum() / target.shape[0]
                
        loss_exp_based = torch.tensor([0.0]).to(loss_prediction.device)
        loss_expl_attn = torch.tensor([0.0]).to(loss_prediction.device)
        context_len = self.explanation_attn_idxs['context'][1] - self.explanation_attn_idxs['context'][0]
        self.explanation_attentions = {'input':{}, 'context':torch.zeros(len(batch['text']), context_len)}
        explanation_gt_input = batch['explanation']
        explanation_gt_context = batch['explanation_context']
        if torch.any(attention_weights<0):
            open('attention_weights_negative.txt','w').write(str(attention_weights.min().item())+'\n')
        ## The last element is output attending to output which can be just whatever
        explanation_probs = 1-torch.exp(-attention_weights)[:,:-1]
        explanation_probs_gt = torch.zeros_like(explanation_probs)
        last_input_idx = 0
        for space in explanation_gt_input.keys():
            i1, i2 = self.explanation_attn_idxs['inputs'][space]
            last_input_idx = max(last_input_idx, i2)
            explanation_gt_space = explanation_gt_input[space].view(explanation_gt_input[space].shape[0], -1).to(explanation_probs.device)
            explanation_probs_gt[:,i1:i2] = explanation_gt_space
            explanation_probs_space = explanation_probs[:,i1:i2]
            self.explanation_attentions['input'][space] = explanation_probs_space.view(explanation_gt_input[space].shape).detach().clone()
        i1, i2 = self.explanation_attn_idxs['context']
        self.explanation_attentions['context'] = explanation_probs[:,last_input_idx+i1:last_input_idx+i2]
        explanation_probs_gt[:,last_input_idx+i1:last_input_idx+i2] = explanation_gt_context.view(explanation_gt_context.shape[0], -1).to(explanation_probs.device)
        no_explanation_mask = explanation_probs_gt.sum(-1) > 0
        explanation_probs_gt[torch.logical_not(attention_mask[:,:-1])] = 0
        loss_expl_attn = self.expl_att_loss(explanation_probs, explanation_probs_gt)[no_explanation_mask].mean()
        if self.cfg['expl_train_attn']:
            loss_exp_based += loss_expl_attn
        
        if batch['explanation_based_data'] is not None: 
            action_probs_pos = nn.Softmax(dim=-1)(self.step(batch['explanation_based_data']['data_pos'])[0])
            action_probs_ref = nn.Softmax(dim=-1)(self.step(batch['explanation_based_data']['data_ref'])[0])
            prob_diff = (- action_probs_pos + action_probs_ref)
            target_exp_based = batch['explanation_based_data']['data_pos']['assistance'].to(action_probs_pos.device).float()
        
        results = {
            'loss': loss_prediction + (loss_exp_based * self.cfg['loss_explanation_weight']),
            'loss_prediction': loss_prediction,
            'loss_exp_based': loss_exp_based,
            'loss_expl_attn': loss_expl_attn,
            'accuracy': accuracy,
            'preds': action_logits.argmax(-1),
            'gts': batch['assistance'].to(action_logits.device).float(),
            'confidences': nn.Softmax(dim=-1)(action_logits).max(-1).values,
        }
        
        return results

    def step(self, batch, introduce_input_gradients=False):
        if 'input_concepts' in batch.keys():
            if self.cfg['concept_src'] == 'no_concepts':
                input_features, input_mask = self.construct_inputs(batch['input_lm_embeds'])
            else:
                input_features, input_mask = self.construct_inputs(batch['input_concepts'], introduce_input_gradients=introduce_input_gradients, from_concepts=True)
        elif 'inputs' in batch.keys():
            raise NotImplementedError("Getting input similarities within the model is out of date!")
            input_features, input_mask = self.construct_inputs(batch['inputs'], introduce_input_gradients=introduce_input_gradients)
        context_features, context_mask = self.construct_context(batch['context'], introduce_input_gradients=introduce_input_gradients)

        output_emb = nn.functional.normalize(self.role_embeddings(self.roles['output']), dim=-1).unsqueeze(0).unsqueeze(0).repeat(input_features.shape[0],1,1)

        output_emb = output_emb.repeat(1,1,3)

        features_in = torch.cat([input_features, context_features, output_emb], dim=1)
        mask_in = torch.cat([input_mask, context_mask, torch.zeros_like(input_mask[:,:1]).to(input_mask.device)], dim=1)
        assert features_in.shape[1] == mask_in.shape[1], f"Features and mask shape mismatch: {features_in.shape} v.s. {mask_in.shape}"

        ## Get transformed feature at output_embed position
        features_out, attention_weights = self.predictor(features_in, src_key_padding_mask=mask_in, return_attention_weights=True)
        features_out = features_out[:,-1,:]
        attention_weights = attention_weights[:,-1,:]
        action_logits = self.action_predictor(features_out)
        
        return action_logits, attention_weights, (mask_in!=-float('inf'))

    def training_step(self, batch, batch_idx):
        results = self(batch, introduce_input_gradients=True)
        
        self.log('Train loss',results['loss'])
        self.log('Train loss prediction',results['loss_prediction'])
        self.log('Train loss explanation',results['loss_exp_based'])
        self.log('Train loss explanation attention',results['loss_expl_attn'])
        self.log('Train accuracy assistance',results['accuracy'])
        
        return results
        
    def validation_step(self, batch, batch_idx):
        results = self(batch, validation=True)
        self.log('Val_ES_accuracy_assistance',results['accuracy'])
        return results['loss']

    def test_step(self, batch, batch_idx):
        self.explanation_attn_idxs = {}
        self.initialize_explanations(batch)
        results = self(batch, validation=True)
        self.spell_out_explanations(batch, 'expl_attn_based', self.explanation_attentions)
        self.log('Test loss assistance',results['loss'])
        self.log('Test accuracy assistance',results['accuracy'])
        for i_act,text_info in enumerate(batch['text']):
            act_name = self.key_from_text(text_info)
            self.explanations[act_name]['ground_truth_check'] = results['gts'].max(-1).indices[i_act]
            self.explanations[act_name]['prediction_check'] = results['preds'][i_act]
        return results['loss']

    def key_from_text(self, text):
        key = ''
        key += text['input']['action'][0]+'_'
        key += '_'.join(text['context'])+'_'
        key = key.replace(' ','_')
        return key

    def initialize_explanations(self, batch):
        for i_act,text_info in enumerate(batch['text']):
            act_name = self.key_from_text(text_info)
            if act_name not in self.explanations:
                self.explanations[act_name] = {
                    'ground_truth':None,
                    'prediction':None,
                    'ground_truth_check':None,
                    'prediction_check':None,
                    'result':None,
                    'confidence':None,
                    'num_precedents':None,
                    'explanation':batch['explanation_text'][i_act]+[[e,True] for e in batch['explanation_context_text'][i_act]],
                    'expl_attn_based': [],
                    'expl_output_based':[], 
                    'expl_gt_based':[]
                    }

    def spell_out_explanations(self, batch, result_type, explanation_weights=None):
        for i_act,text_info in enumerate(batch['text']):
            act_name = self.key_from_text(text_info)
            if len(self.explanations[act_name][result_type]) > 0:
                print(f"!!!!!!!!!!!!!!!! Explanations already present for {act_name} {result_type}")
        for key in self.concept_hierarchy['input'].keys():
            for i_con,con in enumerate(self.concept_hierarchy['input'][key]):
                for i_act,text_info in enumerate(batch['text']):
                    act_name = self.key_from_text(text_info)
                    for i_item,item in enumerate(text_info['input'][key]):
                        self.explanations[act_name][result_type].append((key, item, con, batch['input_concepts'][key]['features'][i_act, i_item, i_con].item(), explanation_weights['input'][key][i_act, i_item, i_con].item()))
        for i_val,val in enumerate(self.concept_hierarchy['context']):
            for i_act,text_info in enumerate(batch['text']):
                    act_name = self.key_from_text(text_info)
                    self.explanations[act_name][result_type].append(('c', 'c', val, batch['context']['features'][i_act, i_val].item(), explanation_weights['context'][i_act, i_val].item()))

    def eval_with_expl(self, batch, batch_idx, action_vocab=None):
        self.eval()
        if action_vocab is None: action_vocab = [i for i in range(self.cfg['num_assistive_actions'])]
        self.initialize_explanations(batch)
        self.to('cuda')
        def to_device(batch):
            for k,v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to('cuda')
                if isinstance(v, dict):
                    batch[k] = to_device(v)
            return batch
        batch = to_device(batch)
        self.automatic_optimization = False
        with torch.enable_grad():
            for result_type in ['expl_output_based', 'expl_gt_based']:
                self.explanation_gradients = {}
                self.optimizers().zero_grad()
                output_as_target = (result_type == 'expl_output_based')
                result = self(batch, introduce_input_gradients=True, output_as_target=output_as_target)
                loss = result['loss']
                self.manual_backward(loss, retain_graph=True)
                explanation_weights = {'input':{k:-v.grad for k,v in self.explanation_gradients['inputs'].items()},
                                       'context':-self.explanation_gradients['context'].grad}
                self.spell_out_explanations(batch, result_type, explanation_weights)
            for i_act,text_info in enumerate(batch['text']):
                act_name = self.key_from_text(text_info)
                self.explanations[act_name]['ground_truth'] = action_vocab[result['gts'].max(-1).indices[i_act]]
                if not isinstance(self.explanations[act_name]['ground_truth_check'], str):
                    self.explanations[act_name]['ground_truth_check'] = action_vocab[self.explanations[act_name]['ground_truth_check']]
                    assert self.explanations[act_name]['ground_truth_check'] == self.explanations[act_name]['ground_truth'], f"Ground truth mismatch: {self.explanations[act_name]['ground_truth_check']} v.s. {self.explanations[act_name]['ground_truth']}"
                self.explanations[act_name]['prediction'] = action_vocab[result['preds'][i_act]]
                if not isinstance(self.explanations[act_name]['prediction_check'], str):
                    self.explanations[act_name]['prediction_check'] = action_vocab[self.explanations[act_name]['prediction_check']]
                    assert self.explanations[act_name]['prediction_check'] == self.explanations[act_name]['prediction'], f"Prediction mismatch: {self.explanations[act_name]['prediction_check']} v.s. {self.explanations[act_name]['prediction']}"
                self.explanations[act_name]['num_precedents'] = batch['num_precedents'][i_act].item()
                self.explanations[act_name]['result'] = 'correct' if self.explanations[act_name]['ground_truth'] == self.explanations[act_name]['prediction'] else 'wrong'
                self.explanations[act_name]['confidence'] = result['confidences'][i_act].item()
        self.automatic_optimization = True
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.cfg['lr'])
        return optimizer
    
    def _explanation_in_gt(self, exp, gt_explanations):
        return exp[:3] in [e[:3] for e in gt_explanations] or exp[2] in [e[0] for e in gt_explanations]

    def write_results(self, output_dir, suffix = ''):
        cutoff = 0.5
        self.explanation_human_readable = {}
        for a in self.explanations:
            self.explanation_human_readable[a] = {'ground_truth':self.explanations[a]['ground_truth'], 
                                                  'prediction':self.explanations[a]['prediction'],
                                                  'num_precedents': self.explanations[a]['num_precedents'],
                                                  'result': self.explanations[a]['result'],
                                                  'confidence': self.explanations[a]['confidence'],
                                                  'explanation':self.explanations[a]['explanation'],
                                                  'expl_attn_based': [],
                                                  'expl_attn_based_MRR':0,
                                                  'expl_output_based':[],
                                                  'expl_output_based_MRR':0,
                                                  'expl_gt_based':[],
                                                  'expl_gt_based_MRR':0,
                                                  }
            
            for expl_type in ['expl_output_based', 'expl_gt_based', 'expl_attn_based']:

                self.explanations[a][expl_type].sort(key=lambda x:abs(x[-1]), reverse=True)
            
                grad_sum = sum([abs(x[3]) for x in self.explanations[a][expl_type]])
                grad_sum = grad_sum if grad_sum > 0 else 1
                grad_rem = deepcopy(grad_sum * cutoff)
                rank = 0
                for _,exp in enumerate(self.explanations[a][expl_type]):
                    concept_thresh = 0.2
                    exp_perc = list(exp[:3])
                    if expl_type in ['expl_output_based', 'expl_gt_based']:
                        if exp[3] < concept_thresh and exp[4] > 0: continue
                        if exp[3] > (1-concept_thresh) and exp[4] < 0: continue
                    else:
                        if exp[3] < concept_thresh: continue
                    correct='[-] '
                    if self._explanation_in_gt(exp_perc, self.explanation_human_readable[a]['explanation']):
                        correct='[+] '
                        if self.explanation_human_readable[a][f'{expl_type}_MRR'] == 0:
                            self.explanation_human_readable[a][f'{expl_type}_MRR'] += 1/(rank+1)
                    exp_perc.append('is TRUE' if exp[3] > 0.5 else 'is FALSE')
                    exp_perc.append('{:5.3f}'.format(exp[3]))
                    exp_perc.append('{:10.8f}'.format(exp[4]/grad_sum))
                    if grad_rem > 0 and rank < 10:
                        self.explanation_human_readable[a][expl_type].append(correct+' : '.join(exp_perc))
                    grad_rem -= abs(exp[-1])
                    rank += 1
        

        evals = complete_evals([(a['prediction'], a['ground_truth']) for a in self.explanation_human_readable.values()], do_now_in_outputs=True)
        cm_object = evals['confusion_matrix_obj']
        evals['confusion_matrix_obj'] = evals['confusion_matrix_obj'].get_numbers_dict()
        del evals['episodes']
        mask_episodes_w_expl = [len(r['explanation']) > 0 for r in self.explanation_human_readable.values()]
        evals['MRR'] = {}
        for k in ['expl_output_based', 'expl_gt_based', 'expl_attn_based']:
            key = k.replace('expl_','')
            mrrs_list = [a[f'{k}_MRR'] for a in self.explanation_human_readable.values()]
            mask_correct_preds = [a['prediction'] == a['ground_truth'] for a in self.explanation_human_readable.values()]
            mask_list = mask_episodes_w_expl
            mrr = sum([1 if mrr>=1 else 0 for (mrr,mask) in zip(mrrs_list, mask_list) if mask])/sum(mask_list) if sum(mask_list) > 0 else 0
            evals['MRR'][key] = (mrr, sum(mask_list))
            correct_mask = [a['result']=='correct' for a in self.explanation_human_readable.values()]
            correct_and_with_eval = [c and e for c,e in zip(correct_mask, mask_list)]
            evals['MRR'][key+'_correct'] = sum([1 if mrr>=1 else 0 for (mrr,mask) in zip(mrrs_list, correct_and_with_eval) if mask])/sum(correct_and_with_eval) if sum(correct_and_with_eval) > 0 else 0, sum(correct_and_with_eval)
            wrong_and_with_eval = [(not c) and e for c,e in zip(correct_mask, mask_list)]
            evals['MRR'][key+'_wrong'] = sum([1 if mrr>=1 else 0 for (mrr,mask) in zip(mrrs_list, wrong_and_with_eval) if mask])/sum(wrong_and_with_eval) if sum(wrong_and_with_eval) > 0 else 0, sum(wrong_and_with_eval)
        evals['episodes'] = self.explanation_human_readable
        print(f"Writing assistance outputs to {os.path.join(output_dir, 'assistances.json')}")
        os.makedirs(output_dir, exist_ok=True)
        json.dump(self.explanations, open(os.path.join(output_dir, f'assistances{suffix}.json'), 'w'), indent=4)
        json.dump(evals, open(os.path.join(output_dir, f'assistances_humanread{suffix}.json'), 'w'), indent=4)
        json.dump({'config':self.cfg, 'concept_hierarchy':self.concept_hierarchy}, open(os.path.join(output_dir, 'config_on_evals.json'), 'w'), indent=4)

        return cm_object