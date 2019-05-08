#coding=utf8
import os
import torch

depressionfilename='depression_new.txt'
normalfilename='normal_new.txt'
dp='normal_depression.txt'

def loadMyData(testrate=0.2):
    mydata=[]
    path='myset'
    #lable: 0 for depression, 1 for normal
    with open(os.path.join(path, depressionfilename), encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            line=line.split()
            mydata.append([line, 0])
    with open(os.path.join(path, dp), encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            line=line.split()
            mydata.append([line, 0])
    with open(os.path.join(path, normalfilename), encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            line=line.split()
            mydata.append([line, 1])

    train=[]
    test=[]

    for i in range(len(mydata)):
        if i%(int(100/(100*testrate)))==0:
            test.append(mydata[i])
        else:
            train.append(mydata[i])

    print("mydata len:" + str(len(mydata)))
    print("train len:" + str(len(train)))
    print("test len:" + str(len(test)))
    return train, test

def loaddict():
    path='myset'
    with open(os.path.join(path, 'mydict.txt'), encoding='utf-8',errors='ignore') as f:
        mydict = f.read()
        mydict = eval(mydict)
    return mydict

def paddingData(dict,id,maxlen=150):
    if len(id)>=maxlen:
        return id[:maxlen]
    else:
        while len(id)<maxlen:
            id.append(dict['<padding>'])
        return id

def word2id(dict,data):
    dataid=[]
    datalabel=[]
    for s in data:
        id=[]
        for w in s[0]:
            if w in dict:
                id.append(dict[w])
            else:
                id.append(dict['<unk>'])#unknown
        paddingData(dict,id)
        dataid.append(id)
        datalabel.append(s[1])

    return dataid,datalabel


def get_dataset_iter(args, data_name = "myset"):
    print("Loading data...")
    train, test=loadMyData()

    args.lenTest=len(test)

    print("Building vocabulary...")
    mydict=loaddict()
    trainid,trainlabel=word2id(mydict,train)
    testid,testlabel=word2id(mydict,test)

    trainid=torch.Tensor(trainid).long()
    trainlabel=torch.Tensor(trainlabel).long()
    testid=torch.Tensor(testid).long()
    testlabel=torch.Tensor(testlabel).long()

    trainSet=torch.utils.data.TensorDataset(trainid,trainlabel)
    testSet=torch.utils.data.TensorDataset(testid,testlabel)

    train_iter=torch.utils.data.DataLoader(trainSet,batch_size=args.batch_size,shuffle=True)
    test_iter = torch.utils.data.DataLoader(testSet, batch_size=args.batch_size, shuffle=False)

    args.embed_num=23461+2
    args.embed_dim=100
    args.class_num=2

    return train_iter, test_iter



def loadlist():
    path='myset'
    with open(os.path.join(path, 'mylist.txt'), encoding='utf-8', errors='ignore') as f:
        mylist = f.read()
        mylist = eval(mylist)
    return mylist

def id2word(s):
    l=loadlist()
    sentence=''
    for i in range(s.size()[0]):
        if s[i]!=0:
            sentence=sentence+' '+l[int(s[i])]
    print(sentence)