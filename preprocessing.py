import os
import pandas as pd
import numpy as np

def process_snli(chaos_path, out_path):
    CHAOS_PATH =  chaos_path
    chaos_snli = pd.read_json(CHAOS_PATH, lines=True)
    modes = ['train', 'dev', 'test']
    dfs = []
    for mode in modes:
        # SNLI_PATH = f'./data/snli_1.0/snli_1.0_{mode}.jsonl'
        SNLI_PATH = f'./data/snli/snli_1.0_{mode}.jsonl'
        # if mode == 'dev': # unify names
        #     mode = 'val'
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

def process_mnli(chaos_path, out_path):
    CHAOS_PATH =  chaos_path
    chaos_mnli = pd.read_json(CHAOS_PATH, lines=True)
    modes = ['train', 'dev_matched', 'dev_mismatched']
    dfs = []
    for mode in modes:
        # SNLI_PATH = f'./data/snli_1.0/snli_1.0_{mode}.jsonl'
        MNLI_PATH = f'./data/mnli/multinli_1.0_{mode}.jsonl'
        # if mode == 'dev_matched': # unify names
        #     mode = 'val'
        mnli = pd.read_json(MNLI_PATH, lines=True)
        # choose columns
        mnli = mnli[['annotator_labels', 'gold_label', 'pairID', 'sentence1', 'sentence2']].copy()
        mnli = mnli.rename(columns={'sentence1': 'premise', 'sentence2': 'hypothesis'})
        # remove chaosnli
        mnli = mnli[~mnli.pairID.isin(chaos_mnli.uid)].copy()
        mnli['label_count'] = mnli.apply(lambda x: count_labels(x['annotator_labels']), axis=1)
        mnli['label_dist'] = mnli.apply(lambda x: label_counts_to_dist(x['label_count']), axis=1)
        # choose those that have 5 labels
        mnli = mnli[mnli['label_count'].map(sum) == 5].reset_index(drop=True).copy()
        if mode != 'dev_mismatched':
            dfs.append(mnli)
        else:
            mnli.to_csv(os.path.join(out_path, 'mnli_dev_mismatched.csv'), index=False)
    del mnli
    full = pd.concat(dfs, ignore_index=True)
    del dfs
    full.to_csv(os.path.join(out_path, 'mnli_full.csv'), index=False)
    del full
        
def process_chaos(snli_path, mnli_path, out_path):
    chaos_snli = pd.read_json(snli_path, lines=True)
    chaos_mnli = pd.read_json(mnli_path, lines=True)
    df_chaos_full = pd.concat([chaos_snli, chaos_mnli])
    df_chaos_full = df_chaos_full.sample(frac=1, random_state=100).reset_index(drop=True)

    # adding p h columns
    df_chaos_full['premise'] = [x['premise'] for x in df_chaos_full['example']]
    df_chaos_full['hypothesis'] = [x['hypothesis'] for x in df_chaos_full['example']]
    
    train_split = int(len(df_chaos_full) * 0.8)
    val_split = int(len(df_chaos_full) * 0.9)
    df_chaos_train = df_chaos_full[:train_split]
    df_chaos_val = df_chaos_full[train_split:val_split]
    df_chaos_test = df_chaos_full[val_split:]
    
    print(f'Number of fine training examples: {len(df_chaos_train)}')
    print(f'Number of fine val examples: {len(df_chaos_val)}')
    print(f'Number of fine test examples: {len(df_chaos_test)}')
    
    df_chaos_train.to_csv(os.path.join(out_path, 'chaos_train.csv'), index=False)
    df_chaos_val.to_csv(os.path.join(out_path, 'chaos_val.csv'), index=False)
    df_chaos_test.to_csv(os.path.join(out_path, 'chaos_test.csv'), index=False)
    
    
    
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
    CHAOS_S_PATH =  './data/chaosNLI/data/chaosNLI_v1.0/chaosNLI_snli.jsonl'
    OUT_PATH = './data/preprocessed'
    CHAOS_M_PATH =  './data/chaosNLI/data/chaosNLI_v1.0/chaosNLI_mnli_m.jsonl'
    
    if not os.path.exists('./data/preprocessed/snli_full.csv'):
        process_snli(CHAOS_S_PATH, OUT_PATH)
    snli_full = pd.read_csv('data/preprocessed/snli_full.csv')
    
    if not os.path.exists('./data/preprocessed/mnli_full.csv'):
        process_mnli(CHAOS_M_PATH, OUT_PATH)
    mnli_full = pd.read_csv('data/preprocessed/mnli_full.csv')
    
    if not os.path.exists('./data/preprocessed/chaos_train.csv'):
        process_chaos(CHAOS_S_PATH, CHAOS_M_PATH, OUT_PATH)
    
    nli_coarse_full = pd.concat([snli_full, mnli_full], ignore_index=True).sample(frac=1, random_state=100).reset_index(drop=True)
    train_split = int(len(nli_coarse_full) * 0.8)
    nli_coarse_train = nli_coarse_full[:train_split].reset_index(drop=True)
    nli_coarse_val = nli_coarse_full[train_split:].reset_index(drop=True)
    print(f'Number of coarse training examples: {len(nli_coarse_train)}')
    print(f'Number of coarse val examples: {len(nli_coarse_val)}')
    nli_coarse_train.to_csv('data/preprocessed/nli_coarse_train.csv', index=False)
    nli_coarse_val.to_csv('data/preprocessed/nli_coarse_val.csv', index=False)

    