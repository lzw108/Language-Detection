#!/usr/bin/env python
# coding: utf-8
import collections
import os
import random
import time
from tqdm import tqdm
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# from torchtext.legacy import data
from torchtext import data
import numpy as np
from utils.data_process import data_preprocess
from utils.dataset import dataset
from utils.model import TextCNN
from utils.running import Running
from utils.args import args
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def train():
    # Load train_data
    data_path = args.data_path
    Dataset = dataset(args)
    train_data_path = os.path.join(data_path, 'train_process.csv')
    train_loader,valid_loader,len_vocab = Dataset.load_data(train_data_path) # 这块检查下TEXT
    # Model initialization
    model = TextCNN(args.emb_dim, [args.kernel_size1, args.kernel_size2, args.kernel_size3], args.kernel_num, len_vocab).to(device)
    # Training
    for epoch in range(args.epochs):
        running = Running(model, args)
        running.train(train_loader, valid_loader, epoch)

def test():
    # Load vocabulary
    vocabs = torch.load('./data/vocabs')
    # Load test_data
    data_path = args.data_path
    test_data_path = os.path.join(data_path, 'test_process.csv')
    Dataset = dataset(args)
    test_loader,temp,len_vocab = Dataset.load_data_test(test_data_path, vocabs)
    # Load model
    model = TextCNN(args.emb_dim, [args.kernel_size1, args.kernel_size2, args.kernel_size3], args.kernel_num, len_vocab).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=lambda storage, loc: storage))
    # test
    running = Running(model, args)
    running.predict(test_loader)

# Test the sentence entered by the user
def test_single():
    while(1):
        input_sen = input("Please input one sentence(Enter q to exit): ")
        if input_sen == 'q':
            exit()
        else:
            vocabs = torch.load('./data/vocabs')
            Dataset = dataset(args)
            single_tensor,temp,len_vocab = Dataset.load_data_single(input_sen, vocabs)
            model = TextCNN(args.emb_dim, [args.kernel_size1, args.kernel_size2, args.kernel_size3], args.kernel_num, len_vocab).to(device)
            model.load_state_dict(torch.load(args.model_path, map_location=lambda storage, loc: storage))
            running = Running(model, args)
            language, p = running.predict_single(single_tensor)
            for i in range(len(language)):

                print("'" + input_sen + "'"+ ' is in %s with a probability %s'% (language[i], p[i]))

if __name__ == '__main__':
    # set random seed
    setup_seed(seed=args.seed)
    # Split data
    if args.data_process:
        data_preprocess(args.data_path)
    if args.train:
        train()
    if args.test:
        test()
    if args.test_single:
        test_single()
    # train()