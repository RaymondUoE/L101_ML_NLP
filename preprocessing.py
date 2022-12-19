import os
import pandas as pd
import numpy as np

def process_snli(chaos_path, out_path):
    CHAOS_PATH =  './data/chaosNLI/data/chaosNLI_v1.0/chaosNLI_snli.jsonl'
    chaos_snli = pd.read_json(CHAOS_PATH, lines=True)
    modes = ['train', 'dev', 'test']
    dfs = []
    for mode in modes:
        SNLI_PATH = f'./data/snli_1.0/snli_1.0_{mode}.jsonl'
        if mode == 'dev': # unify names
            mode = 'val'
        snli = pd.read_json(SNLI_PATH, lines=True)
        # choose columns
        snli = snli[['annotator_labels', 'gold_label', 'captionID', 'pairID', 'sentence1', 'sentence2']].copy()
        snli = snli.rename(columns={'sentence1': 'premise', 'sentence2': 'hypothesis'})
        # remove chaosnli
        snli = snli[~snli.captionID.isin(chaos_snli.uid)].copy()
        snli['label_count'] = snli.apply(lambda x: count_labels(x['annotator_labels']), axis=1)
        snli['label_dist'] = snli.apply(lambda x: label_counts_to_dist(x['label_count']), axis=1)
        # choose those that have >=3 labels
        snli = snli[snli['label_count'].map(sum) >= 3].reset_index(drop=True).copy()
        dfs.append(snli)
    del snli
    full = pd.concat(dfs, ignore_index=True)
    del dfs
    full.to_csv(os.path.join(out_path, 'snli_full.csv'), index=False)
    del full
        
        
        
        
def label_counts_to_dist(counts):
    a = np.array(counts)
    dist = a / np.sum(a)
    return dist.tolist()

def count_labels(labels):
    e = 0
    n = 0
    c = 0
    for l in labels:
        if l == 'entailment':
            e += 1
        elif l == 'neutral':
            n += 1
        elif l == 'contradiction':
            c += 1
        elif l == '':
            continue
        else:
            raise Exception('Label error')
    return [e, n, c]    
    
            
    
if __name__ == "__main__":
    CHAOS_PATH =  './data/chaosNLI/data/chaosNLI_v1.0/chaosNLI_snli.jsonl'
    OUT_PATH = './data/preprocessed'
    process_snli(CHAOS_PATH, OUT_PATH)

    # mode = ''
    # source = ''
    # SNLI_PATH = f'./data/snli_1.0/snli_1.0_{mode}.jsonl'
    # MNLI_PATH = f'./data/multinli_1.0_{mode}.jsonl'
    # CHAOS_PATH = f'./data/chaosNLI/data/chaosNLI_v1.0/chaosNLI_{source}.jsonl'

    