import argparse
from glob import glob
import os
from tqdm.auto import tqdm
import requests
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path

import random

DATAFILES = 'capture24/P[0-9][0-9][0-9].csv.gz'

def download_capture24(datadir, overwrite=False):
    """ Download and extract the capture-24 dataset """
    if overwrite or not os.path.exists(os.path.join(datadir, 'capture24.zip')):
        url = "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001" + \
              "/download_file?file_format=&safe_filename=capture24.zip&type_of_work=Dataset"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open('capture24.zip', "wb") as file:
                for chunk in tqdm(response.iter_content(chunk_size=1024)):
                    file.write(chunk)
        else:
            raise Exception(f"Failed to download file from {url}")

    capture24dir = os.path.join(datadir, 'capture24')

    if overwrite or len(glob(os.path.join(datadir, DATAFILES))) < 151:
        with zipfile.ZipFile(os.path.join(datadir, "capture24.zip"), "r") as f:
            os.makedirs(capture24dir, exist_ok=True)
            for member in tqdm(f.namelist(), desc="Unzipping"):
                try:
                    f.extract(member, datadir)
                except zipfile.error:
                    pass
    else:
        print(f"Using saved capture-24 data at \"{capture24dir}\".")

    return capture24dir

def extract_windows(data, winsize='10s', testratio=0.25, seed=42):
    data_dict = {k: [] for k in data['label'].unique()}
    tr_X, tr_Y, ts_X, ts_Y = [], [], [], []
    random.seed(seed)
    for t, w in tqdm(data.resample(winsize, origin='start')):

        # Check window has no NaNs and is of correct length
        # 10s @ 100Hz = 1000 ticks
        if w.isna().any().any() or len(w) != 1000:
            continue

        x = w[['x', 'y', 'z']].to_numpy()
        y = w['label'].mode(dropna=False).item()

        data_dict[y].append(x)

    for k in data_dict.keys():
        train_len = int(len(data_dict[k]) * (1 - testratio))
        train_cur = 0
        random.shuffle(data_dict[k])
        for d in data_dict[k]:
            if train_cur < train_len:
                tr_X.append(d)
                tr_Y.append(k)
                train_cur += 1
            else:
                ts_X.append(d)
                ts_Y.append(k)

    tr_X = np.stack(tr_X)
    tr_Y = np.stack(tr_Y)

    ts_X = np.stack(ts_X)
    ts_Y = np.stack(ts_Y)

    return tr_X, tr_Y, ts_X, ts_Y

def main(capture24dir, winsize, testratio, seed):
    Path(f"{capture24dir}/train").mkdir(exist_ok=True)
    Path(f"{capture24dir}/test").mkdir(exist_ok=True)

    anno_label_dict = pd.read_csv(f'{capture24dir}/annotation-label-dictionary.csv', index_col='annotation', dtype='string')
    anno_label_dict['ground_truth'] = 0
    for i, l in enumerate(anno_label_dict['label:Willetts2018'].unique()):
        anno_label_dict.loc[anno_label_dict['label:Willetts2018'] == l, 'ground_truth'] = i

    files = [f for f in os.listdir(capture24dir) if f.endswith('.csv.gz')]
    for f in files:
        print(f'Processing {f}')
        data = pd.read_csv(capture24dir + '/' + f,
            index_col='time',
            parse_dates=['time'],
            dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'string'}
        )
        data['label'] = anno_label_dict['ground_truth'].reindex(data['annotation']).to_numpy()
        print('Label distribution:')
        print(data['label'].value_counts(normalize=True))
        
        tr_X, tr_Y, ts_X, ts_Y  = extract_windows(data, winsize, testratio, seed)

        np.savez(f'{capture24dir}/train/{f[:-7]}.npz', X=tr_X, Y=tr_Y)
        np.savez(f'{capture24dir}/test/{f[:-7]}.npz', X=ts_X, Y=ts_Y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42)
    parser.add_argument('--datadir', '-d', default='')
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('--winsize', default='10s')
    parser.add_argument('--testratio', default=0.25)
    
    args = parser.parse_args()

    capture24dir = download_capture24(args.datadir, args.overwrite)
    main(capture24dir, args.winsize, args.testratio, args.seed)