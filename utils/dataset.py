import torch
from torchtext.legacy import data
# from torchtext import data
import random
import re
class dataset:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.TEXT = data.Field(sequential=True, fix_length=self.args.fix_length)
        self.LABEL = data.Field(sequential=False, use_vocab=False)
    # load train_data
    def load_data(self, data_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        fields = [(None, None),(None, None), ('text',self.TEXT),('label2', self.LABEL)]
        training_data=data.TabularDataset(
            path = data_path,
            format = 'csv',
            fields = fields,
            skip_header = True
        )
        # split data to train and dev
        train_data, valid_data = training_data.split(split_ratio=0.3, random_state = random.seed(42))
        self.TEXT.build_vocab(train_data,min_freq=2)
        self.LABEL.build_vocab(train_data)
        torch.save({"text":self.TEXT.vocab,"label":self.LABEL.vocab},'./data/vocabs')
        train_iterator, valid_iterator = data.BucketIterator.splits(
            (train_data, valid_data),
            batch_size = self.args.batch_size,
            sort_key = lambda x: len(x.text),
            sort_within_batch=True,
            device = device)
        return train_iterator, valid_iterator, len(self.TEXT.vocab)
    # load test_data
    def load_data_test(self, data_path, vocabs):
        self.TEXT.vocab = vocabs['text']
        self.LABEL.vocab = vocabs['label']
        test_field = [(None, None),(None, None), ('text',self.TEXT),('label2', self.LABEL)]
        test_data = data.TabularDataset(
            path=data_path,
            format='csv',
            fields=test_field,
            skip_header=True
        )
        test_iter = data.Iterator(
            test_data,
            batch_size=self.args.test_batch_size,
            device=self.device,
            sort=False,
            sort_within_batch=False,
            repeat=False
        )
        return test_iter,' ', len(self.TEXT.vocab)
    # load single sentence data
    def load_data_single(self, sentence, vocabs):
        self.TEXT.vocab = vocabs['text']
        self.LABEL.vocab = vocabs['label']
        punctuation = """！，？｡＂＃＄％＆＇()＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
        re_punctuation = "[{}]+".format(punctuation)
        sentence = re.sub(re_punctuation, "", sentence)
        # re.sub("[{}]+".format(punctuation), "", sentence.decode("utf-8"))
        tokenized = list(sentence.split(' '))
        for i in range(self.args.fix_length-len(tokenized)):
            tokenized.append('<pad>')
        indexed = [(self.TEXT.vocab.stoi[t]) for t in tokenized] #转换为整数序列
        tensor = torch.LongTensor(indexed).to(self.device)
        tensor = tensor.unsqueeze(1)
        return tensor,' ', len(self.TEXT.vocab)
