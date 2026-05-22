import argparse
from glob import glob
import os
from tqdm.auto import tqdm
import requests
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split

def download_ucihar(datadir):
    """ Download and extract the uci-har dataset """
    url = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open('ucihar.zip', "wb") as file:
            for chunk in tqdm(response.iter_content(chunk_size=1024)):
                file.write(chunk)
    else:
        raise Exception(f"Failed to download file from {url}")

    with zipfile.ZipFile(os.path.join(datadir, "ucihar.zip"), "r") as f:
        for member in tqdm(f.namelist(), desc="Unzipping"):
            try:
                f.extract(member, datadir)
            except zipfile.error:
                pass
    with zipfile.ZipFile(os.path.join(datadir, "UCI HAR Dataset.zip"), "r") as f:
        for member in tqdm(f.namelist(), desc="Unzipping"):
            try:
                f.extract(member, datadir)
            except zipfile.error:
                pass

def main(test_ratio, seed):

    features = []
    with open("./UCI HAR Dataset/features.txt") as file:
        for line in file:
            features.append(line.split()[1])
            
    # Renaming duplicate column names
    names = []
    count = {}
    for feature in features:
        if(features.count(feature) > 1):
            names.append(feature)
    for name in names:
        count[name] = features.count(name)

    for i in range(len(features)):
        if(features[i] in names):
            num = count[features[i]]
            count[features[i]] -= 1;
            features[i] = str(features[i] + str(num))
            

    train_df = pd.read_csv("./UCI HAR Dataset/train/X_train.txt", delim_whitespace=True,names=features)
    train_df['subject_id'] = pd.read_csv("./UCI HAR Dataset/train/subject_train.txt",header=None)
    train_df["activity"] = pd.read_csv("./UCI HAR Dataset/train/y_train.txt", header=None)
    activity = pd.read_csv("./UCI HAR Dataset/train/y_train.txt", header=None)
    label_name = activity.map(lambda x: {1: "WALKING", 2:"WALKING_UPSTAIRS", 3:"WALKING_DOWNSTAIRS", 4:"SITTING", 5:"STANDING", 6:"LYING"}.get(x))
    train_df["activity_name"] = label_name

    test_df = pd.read_csv("./UCI HAR Dataset/test/X_test.txt", delim_whitespace=True, names=features)
    test_df['subject_id'] = pd.read_csv("./UCI HAR Dataset/test/subject_test.txt",header=None)
    test_df["activity"] = pd.read_csv("./UCI HAR Dataset/test/y_test.txt", header=None)
    activity = pd.read_csv("./UCI HAR Dataset/test/y_test.txt", header=None)
    label_name = activity.map(lambda x: {1: "WALKING", 2:"WALKING_UPSTAIRS", 3:"WALKING_DOWNSTAIRS", 4:"SITTING", 5:"STANDING", 6:"LYING"}.get(x))
    test_df["activity_name"] = label_name

    data_df = pd.concat([train_df, test_df], ignore_index=True)
    data_df["activity"] -= 1
    data_df["subject_id"] -= 1

    train_data = test_data = test_df.iloc[0:0]
    for idx in range(30):
        temp = []
        for row in data_df.to_dict('records'):
            if row['subject_id'] == idx:
                temp.append(row)
        temp = pd.DataFrame(temp)
        client_train, client_test = train_test_split(temp, test_size=test_ratio, random_state=seed, stratify=temp["activity"])
        train_data = pd.concat([train_data, client_train], ignore_index=True)
        test_data = pd.concat([test_data, client_test], ignore_index=True)

    train_data.to_csv("./UCI HAR Dataset/train/train.csv", index=False)
    test_data.to_csv("./UCI HAR Dataset/test/test.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42)
    parser.add_argument('--datadir', '-d', default='')
    parser.add_argument('--testratio', default=0.25)
    
    args = parser.parse_args()

    download_ucihar(args.datadir)
    main(args.testratio, args.seed)