import os
import json
import argparse
import torch
from transformers import AutoTokenizer, DistilBertModel
from sentence_transformers import SentenceTransformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
class EmbeddingGenerator():
    def __init__ (self):
        self.map = {'concepts':{}, 'items':{}}

    def _add_embedding_to_map(self, text, text_type):
        if text in self.map[text_type].keys():
            return
        self.map[text_type][text] = self(text).detach().to('cpu')

    def add_concepts(self, textlist):
        for i,text in enumerate(textlist):
            self._add_embedding_to_map(text, 'concepts')

    def add_items(self, textlist):
        for i,text in enumerate(textlist):
            self._add_embedding_to_map(text, 'items')

    def save(self, out_path):
        if '.' not in out_path: out_path = os.path.join(out_path, f"{self.__class__.__name__}.pt")
        torch.save(self.map, out_path)


class DistilBertEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self):
        super().__init__()
        self.textizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(DEVICE)

    def __call__(self, text):
        text_nl = text.replace('_',' ')
        inputs = self.textizer(text=text_nl, return_tensors="pt").to(DEVICE)
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state[0,1:-1,:]
        last_hidden_states = last_hidden_states.mean(dim=0)
        assert last_hidden_states.size() == torch.Size([768])
        return last_hidden_states.detach()


class SentenceTransformerEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self):
        super().__init__()
        self.sentence_model = SentenceTransformer("stsb-roberta-large").to(DEVICE)
        
    def __call__(self, text):
        return self.sentence_model.encode([text], batch_size=1, convert_to_tensor=True, device=DEVICE).detach()

def get_embedder(embedder_type, path=None):
    if embedder_type.lower() == 'sentence':
        embedding_model = SentenceTransformerEmbeddingGenerator()
    elif embedder_type.lower() == 'distilbert':
        embedding_model = DistilBertEmbeddingGenerator()
    else:
        raise NotImplementedError(f"only sentence and distilbert embedders are implemented yet, not {args.embedder.lower()}")
    if path is not None: embedding_model.map = torch.load(path)
    return embedding_model
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model on routines.')
    parser.add_argument('--in_path', type=str, default='data/llama_generations.jsonl')
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--embedder', type=str, default='sentence')

    args = parser.parse_args()
    
    if args.out_path is None:
        args.out_path = args.in_path.replace('.json',f'_{args.embedder}.pt')
    
    embedding_model = get_embedder(args.embedder)
    
    data = [json.loads(l.strip()) for l in open(args.in_path).readlines()]
    items = list(set([d['item'] for d in data]))
    concepts = list(set([d['concept'] for d in data]))

    embedding_model.add_items(items)
    embedding_model.add_concepts(concepts)
    embedding_model.save(args.out_path)
