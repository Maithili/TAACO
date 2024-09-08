from argparse import ArgumentError
import os
import json
import sys
sys.path.append('utils')
sys.path.append('dataloader')
from adict import adict
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from pytorch_lightning import LightningModule
from eval_helpers import complete_evals

from embedders import get_embedder

class ConceptActionMappingModule(LightningModule):
    def __init__(self, concept_name, **kwargs):
        super().__init__()
        self.cfg = kwargs
        self.cfg['label_mean'] = (self.cfg['label_min'] + self.cfg['label_max'])/2
        self.cfg['similarity_scale'] = 2
        self.cfg['label_scale'] = self.cfg['similarity_scale']/(self.cfg['label_max']-self.cfg['label_min'])
        self.cfg['concept_name'] = concept_name

        dim_mid1 = int(round((self.cfg['hidden_concept_dim']*2)))
        dim_mid2 = int(round((self.cfg['hidden_concept_dim']*1.5)))
        dim_mid3 = int(round((self.cfg['hidden_concept_dim']*1)))
        dim_mid4 = int(round((self.cfg['hidden_concept_dim']*0.5)))

        ### Pre-embedding Model ###
        self.classifier0 = nn.Sequential(nn.Linear(self.cfg['input_embed_dim']*2, dim_mid1),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.2))
        self.classifier1 = nn.Sequential(nn.Linear(dim_mid1, dim_mid1),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.2))
        self.classifier2 = nn.Sequential(nn.Linear(dim_mid1, dim_mid2),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.2))
        self.classifier3 = nn.Sequential(nn.Linear(dim_mid2, dim_mid2),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.2))
        self.classifier4 = nn.Sequential(nn.Linear(dim_mid2, dim_mid3),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.2))
        self.classifier5 = nn.Sequential(nn.Linear(dim_mid3, dim_mid3),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.2))
        self.classifier6 = nn.Sequential(nn.Linear(dim_mid3, dim_mid4),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(dim_mid4, 2)
                                        )
        self.concept_embs = None
        self.concept_vocab = None
        self.reset_validation()

    def reset_validation(self):
        self.similarities_generated = {}

    def add_concept_vocab(self, concepts):
        self.concept_vocab = concepts
        self.concept_embs = None
        
    def create_concept_vocab(self, concepts_text):
        embedding_model = get_embedder(self.cfg['embedder'])
        embedding_model.add_concepts(concepts_text)
        self.concept_vocab = {c:{'bert':embedding_model.map['concepts'][c]} for c in concepts_text}
        self.concept_embs = None
        
    @classmethod
    def load_from(cls, dirpath, concept_name=None, config=None, no_vocab=False, concepts_text=None):
        ckpt_file_options = [f for f in os.listdir(dirpath) if f.endswith('.ckpt')]
        def get_metric(ckpt_name, metric='Val_ES_accuracy'):
            items = [c.split('=') for c in ckpt_name.replace('.ckpt','').split('-')]
            acc = [float(c[1]) for c in items if metric in c[0]]
            return acc[0] if len(acc) > 0 else 0
        ckpt_file = sorted(ckpt_file_options, key=get_metric, reverse=True)[0]
        if all([get_metric(ckpt_file, metric='epoch') < 2 for ckpt_file in ckpt_file_options]):
            ckpt_file = 'last.ckpt'
        print(f"  - Loading model from {os.path.join(dirpath, ckpt_file)}")
        if config is None:
            config = json.load(open(os.path.join(dirpath, 'config.json')))
        else:
            config['concept_name'] = concept_name
        model = cls.load_from_checkpoint(os.path.join(dirpath, ckpt_file), **config)
        model.previous_epochs = get_metric(ckpt_file, metric='epoch')
        if not no_vocab:
            concept_file = os.path.join(dirpath, 'concepts.pt')
            model.concept_embs = torch.load(concept_file)
            if 'concepts_in.pt' in os.listdir(dirpath):
                concept_in_file = os.path.join(dirpath, 'concepts_in.pt')
                model.concept_vocab = torch.load(concept_in_file)
        if concepts_text is not None:
            model.create_concept_vocab(concepts_text)

        return model
 
    def save_to(self, dirpath):
        os.makedirs(dirpath, exist_ok=True)
        json.dump(self.cfg, open(os.path.join(dirpath, 'config.json'), 'w'), indent=4)
        torch.save(self.state_dict(), os.path.join(dirpath, 'weights.pt'))
        if self.concept_vocab is not None:
            self.embed_concepts(out_dir=dirpath)
            
    def save_config(self, dirpath):
        os.makedirs(dirpath, exist_ok=True)
        json.dump(self.cfg, open(os.path.join(dirpath, 'config.json'), 'w'), indent=4)
        if self.concept_vocab is not None:
            self.embed_concepts(out_dir=dirpath)

    def embed_concepts(self, out_dir=None):
        if self.concept_embs is None:
            assert self.concept_vocab is not None
            self.concept_embs = {
                'text': [k for k,v in self.concept_vocab.items()],
                'embeddings_bert': torch.cat([v['bert'] for k,v in self.concept_vocab.items()], dim=0),
            }
        if out_dir is not None:
            concept_file = os.path.join(out_dir, 'concepts.pt')
            concept_in_file = os.path.join(out_dir, 'concepts_in.pt')
            print(f"Writing concepts to {concept_file}")
            torch.save(self.concept_embs, concept_file)
            torch.save(self.concept_vocab, concept_in_file)
        return self.concept_embs

    def get_item_concept_similarities(self, items=None, concepts=None):
        if concepts is None:
            concepts = self.concept_embeddings('bert').to(self.device).unsqueeze(0)
        embeddings = items.to(self.device).unsqueeze(1)
        s1 = embeddings.shape[0]
        s2 = concepts.shape[1]
        embeddings_repeat = embeddings.repeat(1,s2,1).view(s1*s2, -1)
        concepts_repeat = concepts.repeat(s1,1,1).view(s1*s2, -1)
        results = self.forward({'items':embeddings_repeat, 'concepts':concepts_repeat, 'labels':None})
        similarity_matrix = results['similarities'].view(s1, s2)
        return similarity_matrix

    def concept_embeddings(self, embedding_type='latent'):
        if embedding_type == 'latent':
            return self.embed_concepts()['embeddings_latent']
        if embedding_type == 'bert':
            return self.embed_concepts()['embeddings_bert']
        if embedding_type == 'onehot':
            return self.embed_concepts()['embeddings_onehot']
        raise ArgumentError(f"Invalid embedding type: {embedding_type} not in ['latent', 'bert', 'onehot']")

    def concept_names(self):
        return self.embed_concepts()['text']
    
    def mlp_step(self, x):
        x1 = self.classifier0(x)
        x2 = self.classifier1(x1) + x1
        x3 = self.classifier2(x2)
        x4 = self.classifier3(x3) + x3
        x5 = self.classifier4(x4)
        x6 = self.classifier5(x5) + x5
        x7 = self.classifier6(x6)
        return x7
        
    def forward(self, batch):
        """
        Args:
        """
        action_embeddings = batch['items']
        concept_embeddings = batch['concepts']

        action_concept_similarities =  self.mlp_step(torch.concatenate([action_embeddings, concept_embeddings], dim=-1))
        desired_similarities = torch.zeros_like(action_concept_similarities)
        total_loss = 0
        loss_action_concept_alignment = 0

        if batch['labels'] is not None:
            desired_similarities = F.one_hot((batch['labels'].float() > 0.5).long(), num_classes=2).float().to(self.device)
            similarity_weight = torch.abs(batch['labels'])
            loss_action_concept_alignment = (torch.nn.CrossEntropyLoss(reduction='none')(action_concept_similarities, desired_similarities) * similarity_weight).mean()

            total_loss += loss_action_concept_alignment

        results = {
            'loss' : total_loss,
            'loss_action_concept_alignment': loss_action_concept_alignment,
            'similarities' : torch.nn.Softmax(dim=-1)(action_concept_similarities)[:,1],
            'similarity_labels' : desired_similarities.argmax(-1),
        }
        
        return results

    def training_step(self, batch, batch_idx):
        results = self(batch)
        self.log(f"Train {self.cfg['concept_name']}: loss",results['loss'])
        self.log(f"Train {self.cfg['concept_name']}: loss_action_concept_alignment",results['loss_action_concept_alignment'])
        episodes = [(simil.item(), label.item()) for simil, label in zip(results['similarities'], results['similarity_labels'])]
        episodes = [(int(round(a)), int(b)) for a,b in episodes]
        evals = complete_evals(episodes)['confusion_matrix_obj'].get_numbers_dict()
        self.log(f"Train {self.cfg['concept_name']}: accuracy",evals['accuracy'])
        self.concept_embs = None
        return results['loss']
        
    def validation_step(self, batch, batch_idx):
        results = self(batch)
        self.log(f"Val {self.cfg['concept_name']}: loss",results['loss'])
        self.reset_validation()
        episodes = [(simil.item(), label.item()) for simil, label in zip(results['similarities'], results['similarity_labels'])]
        episodes = [(int(round(a)), int(b)) for a,b in episodes]
        evals = complete_evals(episodes)['confusion_matrix_obj'].get_numbers_dict()
        self.log(f"Val {self.cfg['concept_name']}: accuracy",evals['accuracy'])
        self.log(f"Val_ES_accuracy_{self.cfg['concept_name']}",evals['accuracy'])
        return 

    def test_step(self, batch, batch_idx):
        results = self(batch)
        self.log(f"Test {self.cfg['concept_name']}: loss",results['loss'])
        for action,concept,simil,label in zip(batch['items_text'], batch['concepts_text'], results['similarities'], results['similarity_labels']):
            if concept not in self.similarities_generated: self.similarities_generated[concept] = {} 
            self.similarities_generated[concept][action] = (simil.item(), label.item())
        episodes = [(simil.item(), label.item()) for simil, label in zip(results['similarities'], results['similarity_labels'])]
        episodes = [(int(round(a)), int(b)) for a,b in episodes]
        evals = complete_evals(episodes)['confusion_matrix_obj'].get_numbers_dict()
        self.log(f"Test {self.cfg['concept_name']}: accuracy",evals['accuracy'])
        return 

    def write_results(self, output_dir, file_prefix=''):
        print(f"Writing generations to {os.path.join(output_dir, f'{file_prefix}generations.json')}")
        os.makedirs(output_dir, exist_ok=True)
        results_generated = {}
        episodes = []
        for concept, results in self.similarities_generated.items():
            episodes += [(int(round(a)), int(b)) for a,b in results.values()]
            results_generated[concept] = complete_evals([(int(round(a)), int(b)) for a,b in results.values()])
            results_generated[concept]['confusion_matrix_obj'] = results_generated[concept]['confusion_matrix_obj'].get_numbers_dict()
        evals = complete_evals(episodes)
        cm_object = evals['confusion_matrix_obj']
        evals['confusion_matrix_obj'] = evals['confusion_matrix_obj'].get_numbers_dict()
        
        os.makedirs(output_dir, exist_ok=True)
        json.dump(self.similarities_generated, open(os.path.join(output_dir, f'{file_prefix}generations.json'), 'w'), indent=4)
        json.dump({'overall':evals, 'concept-wise':results_generated}, open(os.path.join(output_dir, f'{file_prefix}results.json'), 'w'), indent=4)
        return cm_object

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.cfg['lr_pretrain'], weight_decay=self.cfg['weight_decay'])
