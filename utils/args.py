import argparse

parser = argparse.ArgumentParser()

# file path
parser.add_argument("--data_path", type=str, default="./data/")
parser.add_argument("--test_size", type=float, default=0.1)
parser.add_argument("--model_path", type=str, default="./model/cnn_model_paras.pkl")
parser.add_argument("--seed", type=int, default=42)
# parameter
parser.add_argument("--num_classes", type=int, default=403)
parser.add_argument("--topk", type=int, default=3)
#
parser.add_argument("--step_size", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--test_batch_size", type=int, default=1024)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--fix_length", type=int, default=100)

# model
parser.add_argument("--emb_dim", type=int, default=50)
parser.add_argument("--kernel_num", type=int, default=60)

parser.add_argument("--kernel_size1", type=int, default=3)
parser.add_argument("--kernel_size2", type=int, default=5)
parser.add_argument("--kernel_size3", type=int, default=4)
# parser.add_argument("--kernel_size4", type=int, default=6)

parser.add_argument("--data_process", action='store_true',
                    help="Whether to run training.")
parser.add_argument("--train", action='store_true',
                    help="Whether to run training.")
parser.add_argument("--test", action='store_true',
                    help="Whether to run training.")
parser.add_argument("--test_single", action='store_true',
                    help="Whether to run training.")

args = parser.parse_args()
