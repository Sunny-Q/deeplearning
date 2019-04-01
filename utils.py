import argparse

def get_common_args():
    parser = argparse.ArgumentParser()
    # MODEL main里面选择
    #parser.add_argument('--model_name', type=str, default='TextCNN', help='the model', required=False)

    # common args
    parser.add_argument('--cuda', type=bool, default=True, help='enable the cuda', required=False)
    parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to put training data', required=False)
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models', required=False)
    parser.add_argument('--save_model_name', type=str, default='best_model.pth', help='Directory to save models', required=False)
    parser.add_argument('-log-interval', type=int, default=1, help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')

    # models args
    parser.add_argument("-static", type=bool, default=False, help="fix the embedding [default: False]")
    parser.add_argument("-add_pretrain_embedding", type=bool, default=True)  #有已经训练好的embedding.txt文件
    parser.add_argument("-pretrain_embedding_path", type=str, default="myset/embedding.txt")  # 有已经训练好的embedding.txt文件
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate", required=False)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size", required=False)
    parser.add_argument('--epochs', type=int, default=256, help='Number of training epochs', required=False)
    parser.add_argument('--hidden_dim', type=int, default=128, help='the hidden size', required=False)
    parser.add_argument('--embed_dim', type=int, default=128, help='the embedding dim of word embedding', required=False)  #原128，想改100
    parser.add_argument('--pretrain', type=bool, default=True, help='enable the pretrain embedding', required=False)

    # CNN args
    parser.add_argument('--kernel_num', type=int, default=100, help='Number of cnn kernel', required=False)
    parser.add_argument('-kernel_sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')

    # LSTM args
    parser.add_argument('--lstm_hidden_dim', type=int, default=100, help='Number of lstm hidden dim', required=False)
    parser.add_argument('--lstm_num_layers', type=int, default=1, help='Number of lstm layer numbers')
    args, unknown = parser.parse_known_args()
    return args, unknown
