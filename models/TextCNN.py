#coding=utf8
from torchtext import data
from models.BasicModule import *
import gensim
import torch
class TextCNN(BasicModule):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args
        # self.pretrained_embeddings = pretrained_embeddings
        self.embed_num = self.args.embed_num
        self.embed_dim = self.args.embed_dim # 128
        self.class_num = self.args.class_num # 2
        self.input_dim = 1
        self.output_dim = args.kernel_num # 100
        self.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

        self.embed = nn.Embedding(self.embed_num, self.embed_dim, padding_idx=0)
        if args.add_pretrain_embedding:
            pretrained_embeddings = np.loadtxt(args.pretrain_embedding_path)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            #model = gensim.models.KeyedVectors.load_word2vec_format('embedding.txt', binary=False)
            #weight = torch.FloatTensor(model.syn0)
            #self.embedding = nn.Embedding.from_pretrained(weight)
        self.embed.weight.requires_grad = self.args.static != True
        self.convs1 = nn.ModuleList([nn.Conv2d(self.input_dim,
                                               self.output_dim,
                                               (K, self.embed_dim))
                                     for K in self.kernel_sizes])
        self.dropout = nn.Dropout(args.dropout)

        self.fc1 = nn.Linear(len(self.kernel_sizes) * self.output_dim, self.class_num)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.relu(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, inputs_length):
        x = self.embed(x)
        x = x.unsqueeze(1)

        # convolution
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc1(x)
        return logit



def get_trainable_param_num(model):
    """ get the number of trainable parameters

    Args:
        model:

    Returns:

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_param_num(model):
    """ get the number of parameters

    Args:
        model:

    Returns:

    """
    return sum(p.numel() for p in model.parameters())

