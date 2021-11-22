import torch
from torch import nn
import torch.nn.functional as F
from utils.args import args

# TextCNN
class TextCNN(nn.Module):
    def __init__(self, emb_dim, kernel_sizes, kernel_num, len_vocab):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(len_vocab, emb_dim)
        self.embedding_dropout = nn.Dropout(0.5)
        # Use nn.ModuleList to install three nn.Sequential convolutional blocks
        self.convs = nn.ModuleList([
            # 每个卷积块装有一层卷积和LeakyReLU激活函数
            nn.Sequential(
                nn.Conv1d(in_channels=emb_dim, out_channels=kernel_num, kernel_size=size),
                nn.LeakyReLU()
            )
            for size in kernel_sizes
        ])
        in_features = kernel_num * len(kernel_sizes)
        self.linear1 = nn.Linear(in_features=in_features, out_features=in_features//2)
        self.fc_dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(in_features=in_features//2, out_features=args.num_classes)

    def forward(self,x):
        # (length, batch_size)
        out = self.embedding(x)
        out = self.embedding_dropout(out)
        # (length, batch_size, emb) -> (batch_size, emb, length)
        out = torch.transpose(out, 1, 2)
        out = torch.transpose(out, 0, 2)
        out = [conv(out) for conv in self.convs]

        out = [F.max_pool1d(one, kernel_size=one.size(2), stride=2) for one in out]
        # Concatenate dimension 1, and remove dimension 2
        out = torch.cat(out, dim=1).squeeze(2)
        out = self.linear1(F.leaky_relu(out))
        out = self.fc_dropout(out)
        out = self.linear2(F.leaky_relu(out))
        return out
