import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split

def unzip_data():
    with zipfile.ZipFile('./data/raw/filtered_paranmt.zip', 'r') as zip_ref:
        zip_ref.extractall('./data/raw/')

 
def load_data():
    data = pd.read_csv('./data/raw/filtered.tsv', sep='\t', index_col=0)
    return data


def preprocess(data):
    print("Data processing...")
    
    for i, row in data.iterrows():
        if row['ref_tox'] < row['trn_tox']:
            data.at[i, 'reference'] = row['translation']
            data.at[i, 'translation'] = row['reference']
            
            data.at[i, 'ref_tox'] = row['trn_tox']
            data.at[i, 'trn_tox'] = row['ref_tox']


def filter(data):
    data = data[(data['ref_tox'] > 0.85) & (data['trn_tox'] < 0.15) & (data['similarity'] > 0.68)]
    data.reset_index(drop=True, inplace=True)
    return data

            
def make_dataset():
    # Unzip dataset
    unzip_data()
    
    # Load dataset
    data = load_data()

    # Preprocess data (distribute toxicity levels)
    preprocess(data)

    # Filter data
    data = filter(data)

    # Retrieve reference and translation
    data = data[['reference', 'translation']]

    # Split train and test data
    train_data, test_data = train_test_split(data)

    # Save the data
    train_data.to_csv('./data/interim/train.csv', index=False)
    test_data.to_csv('./data/interim/test.csv', index=False)

if __name__ == "__main__":
    make_dataset()
    print("Script is completed! The data is saved in ./data/interim")
