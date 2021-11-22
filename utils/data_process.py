import pandas as pd
from sklearn.model_selection import train_test_split
from utils.args import args
import numpy as np
def data_preprocess(data_path):
    # read data
    df = pd.read_csv(data_path+'./sentences.csv',delimiter='\t')
    # df = pd.read_csv(data_path+'./sentences.csv',delimiter=',')
    df.loc[-1] = df.columns.tolist()
    df.index = df.index + 1
    df.sort_index(inplace=True)
    # add head
    df.columns = ['id', 'label', 'text']
    # Count labels and convert them to integer values
    status_dict = df['label'].unique().tolist()
    # print(status_dict)
    np.save(data_path + 'language.npy',status_dict,)
    df['label2']=df['label'].apply(lambda x : status_dict.index(x))
    # Split data, the training data will be divided into training set and
    # validation set in the dataset script.
    train, valid = train_test_split(df, test_size = args.test_size, random_state=0)
    train.to_csv(data_path + 'train_process.csv', index=False)
    valid.to_csv(data_path + 'test_process.csv', index=False)