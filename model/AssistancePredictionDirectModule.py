import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from AssistancePredictionModule import AssistancePredictionModule


class AssistancePredictionDirectModule(AssistancePredictionModule):
    def __init__(self, all_concepts, **kwargs):
        super().__init__(all_concepts, **kwargs)
        self.item_projector = nn.Linear(self.cfg['concept_emb_input_dim'], self.cfg['latent_map_dim_assistance_concept']*2)

        
    def construct_inputs(self, batch_inputs, introduce_input_gradients=False, from_concepts=True):
        features_encoded = []
        mask = []
        self.explanation_gradients['inputs'] = {}
        
        for key,data in batch_inputs.items():
            batch, num_seq, _ = data['features'].shape
            values = self.item_projector(data['features'])            
            roles = self.role_embeddings(self.roles[key]).unsqueeze(0).unsqueeze(1).repeat(batch,num_seq,1)
            values_norm = nn.functional.normalize(values, dim=-1)
            roles_norm = nn.functional.normalize(roles, dim=-1)
            features_encoded.append(torch.cat([values_norm, roles_norm], dim=-1))
            mask.append(data['mask'].view(batch,-1))
        
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
        
        loss_attn_sparsity = torch.tensor([0.0]).to(loss_prediction.device)
        
        loss_exp_based = torch.tensor([0.0]).to(loss_prediction.device)
        loss_expl_cf = torch.tensor([0.0]).to(loss_prediction.device)
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
            explanation_gt_space = explanation_gt_input[space].sum(-1).to(explanation_probs.device)
            explanation_probs_gt[:,i1:i2] = explanation_gt_space.detach().clone()
            explanation_probs_space = explanation_probs[:,i1:i2]
            self.explanation_attentions['input'][space] = explanation_probs_space.view(explanation_gt_space.shape).detach().clone()
        i1, i2 = self.explanation_attn_idxs['context']
        self.explanation_attentions['context'] = explanation_probs[:,last_input_idx+i1:last_input_idx+i2]
        explanation_probs_gt[:,last_input_idx+i1:last_input_idx+i2] = explanation_gt_context.view(explanation_gt_context.shape[0], -1).to(explanation_probs.device)

        no_explanation_mask = (explanation_probs_gt.sum(-1) > 0)
        explanation_probs_gt[torch.logical_not(attention_mask[:,:-1])] = 0
        # if not validation:
        #     loss_expl_attn = self.expl_att_loss(explanation_probs, explanation_probs_gt)[no_explanation_mask].mean()
        # else:
        log_prob_of_gt = torch.log(explanation_probs)
        log_prob_of_gt[explanation_probs_gt < 0.5] = torch.log((1-explanation_probs))[explanation_probs_gt < 0.5]
        loss_expl_attn = log_prob_of_gt.mean()
        if self.cfg['expl_train_attn']:
            loss_exp_based += loss_expl_attn
        
        results = {
            'loss': loss_prediction + (loss_exp_based * self.cfg['loss_explanation_weight']),
            'loss_prediction': loss_prediction,
            'loss_exp_based': loss_exp_based,
            'loss_expl_attn': loss_expl_attn,
            'loss_expl_cf': loss_expl_cf,
            'loss_attn_sparsity': loss_attn_sparsity,
            'accuracy': accuracy,
            'preds': action_logits.argmax(-1),
            'gts': batch['assistance'].to(action_logits.device).float(),
            'confidences': nn.Softmax(dim=-1)(action_logits).max(-1).values,
        }
        
        return results
    
    def spell_out_explanations(self, batch, result_type, explanation_weights=None):
        for i_act,text_info in enumerate(batch['text']):
            act_name = self.key_from_text(text_info)
            if len(self.explanations[act_name][result_type]) > 0:
                print(f"!!!!!!!!!!!!!!!! Explanations already present for {act_name} {result_type}")
        for key in self.concept_hierarchy['input'].keys():
            for i_act,text_info in enumerate(batch['text']):
                act_name = self.key_from_text(text_info)
                for i_item,item in enumerate(text_info['input'][key]):
                    self.explanations[act_name][result_type].append((key, item, 'Concept', 1, explanation_weights['input'][key][i_act, i_item].item()))
        for i_val,val in enumerate(self.concept_hierarchy['context']):
            for i_act,text_info in enumerate(batch['text']):
                    act_name = self.key_from_text(text_info)
                    self.explanations[act_name][result_type].append(('c', 'c', val, batch['context']['features'][i_act, i_val].item(), explanation_weights['context'][i_act, i_val].item()))


    def _explanation_in_gt(self, exp, gt_explanations):
        return exp[:2] in [e[:2] for e in gt_explanations] or exp[2] in [e[0] for e in gt_explanations]

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
        for result_type in ['expl_output_based', 'expl_gt_based']:
            self.explanation_gradients = {}
            explanation_weights = {'input':{k:torch.zeros_like(v['mask']) for k,v in batch['input_concepts'].items()},
                                    'context':torch.zeros_like(batch['context']['mask'])}
            self.spell_out_explanations(batch, result_type, explanation_weights)
        for i_act,text_info in enumerate(batch['text']):
            act_name = self.key_from_text(text_info)
            self.explanations[act_name]['ground_truth_check'] = action_vocab[self.explanations[act_name]['ground_truth_check']]
            self.explanations[act_name]['ground_truth'] = self.explanations[act_name]['ground_truth_check']
            self.explanations[act_name]['prediction_check'] = action_vocab[self.explanations[act_name]['prediction_check']]
            self.explanations[act_name]['prediction'] = self.explanations[act_name]['prediction_check']
            self.explanations[act_name]['num_precedents'] = batch['num_precedents'][i_act].item()
            self.explanations[act_name]['result'] = 'correct' if self.explanations[act_name]['ground_truth'] == self.explanations[act_name]['prediction'] else 'wrong'
            self.explanations[act_name]['confidence'] = 1
        self.automatic_optimization = True
    