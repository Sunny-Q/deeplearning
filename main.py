import sys
import torch
from mydatasets2 import *
from models.LSTMSelfAttention import *
from models.TextCNN import *
from utils import *


def validate(model, val_iter, args,inputs_length=150):
    model.eval()
    corrects, avg_loss = 0.0, 0.0
    for (inputs,target) in val_iter:
        if args.cuda and args.device != -1:
            #inputs, inputs_length, target = inputs.cuda(), inputs_length.cuda(), target.cuda()
            inputs, target = inputs.cuda(), target.cuda()

        logit = model(inputs, inputs_length)
        loss = F.cross_entropy(logit, target, size_average=False)
        correct = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

        avg_loss += loss.item()
        corrects += correct

    #size = len(val_iter.dataset)
    size=args.lenTest
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) '.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))


def train(model, train_iter, val_iter, args,inputs_length=150):
    print("begin to train models...")
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    steps = 0
    for epoch in range(1, args.epochs + 1):
        for (inputs,target) in train_iter:
            if args.cuda and args.device != -1:
                #inputs, inputs_length, target = inputs.cuda(), inputs_length.cuda(), target.cuda()
                inputs, target = inputs.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(inputs, inputs_length)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

                accuracy = 100*corrects/target.size(0)
                sys.stdout.write(
                    '\rBatch[E{}S{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(epoch,steps,
                                                                             loss.item(),
                                                                             accuracy,
                                                                             corrects,
                                                                             target.size(0)))
            #import time
            #time.sleep(1)
            if steps % args.test_interval == 0:
                validate(model, val_iter, args)


def mytest(model,val_iter,args,inputs_length=150):
    model.eval()
    corrects, avg_loss = 0.0, 0.0
    j = 0
    for (inputs,target) in val_iter:
        if args.cuda and args.device != -1:
            #inputs, inputs_length, target = inputs.cuda(), inputs_length.cuda(), target.cuda()
            inputs, target = inputs.cuda(), target.cuda()

        logit = model(inputs, inputs_length)
        loss = F.cross_entropy(logit, target, size_average=False)
        correct = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

        #print(torch.max(logit, 1)[1].view(target.size()).data)
        #print(target.data)


        l=int(target.size()[0])
        for i in range(l):
            if torch.max(logit, 1)[1].view(target.size()).data[i]!=target.data[i]:
                #print(inputs.data[i])
                print(int(target.data[i]))
                id2word(inputs.data[i])



if __name__ == '__main__':

    args, unknown = get_common_args()
    # -1 means cpu, else the gpu index 0,1,2,3
    args.device = -1
    print("args : " + str(args))
    print("unknown args : " + str(unknown))

    model_list = ['TextCNN', 'LSTMSelfAttention']
    args.model_name = model_list[0]

    Train=False

    if Train:
        train_iter, val_iter = get_dataset_iter(args, "myset")  # 这里要改成自己的！！！！！！！！！！！！！！
        model = eval(args.model_name)(args)
        if args.cuda and args.device != -1:
            torch.cuda.set_device(args.device)
            model = model.cuda()
        train(model, train_iter, val_iter, args)
        model.save()
    else:
        args.embed_num = 12414 + 2
        args.embed_dim = 100
        args.class_num = 2

        train_iter, val_iter = get_dataset_iter(args, "myset")
        model = eval(args.model_name)(args)

        if args.cuda and args.device != -1:
            torch.cuda.set_device(args.device)
            model = model.cuda()

        filename = '0331_21_54_12.pth'
        loadPath = os.path.join('textcnn', filename)
        model.load(loadPath)

        print('loading ok')
        mytest(model,val_iter,args)
        #validate(model,val_iter,args)



