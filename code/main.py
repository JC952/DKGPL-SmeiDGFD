import argparse
import logging
import os
from datetime import datetime
from data.construct_loader import Fault_dataset
from utils.SetSeed import set_random_seed
from utils.logger import setlogger, result_log
from utils.train_test import train_test

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--dataset_name', type=str, default="GearBox.BJUT", help='name of dataset')
    parser.add_argument('--target_id', type=str, default='2400', help='target domain')
    parser.add_argument('--source_id_list', type=int, nargs='+', default=[1200, 1800, 3000], help='List of source domain IDs')
    parser.add_argument('--data_ratio', type=int, default=0.5, help='percentage of dataset division')
    parser.add_argument('--miss_class', nargs='+', type=int, default=[], help='deleting labels from a class')
    parser.add_argument('--FFT', type=bool, default=True, help='whether to Fourier transform the data')
    parser.add_argument('--normalize_type', type=str, default='mean-std', help='data normalization methods')
    parser.add_argument('--model_name', type=str, default='M4', help='the name of the method')
    parser.add_argument('--lr', type=float, default= 0.0001, help='the learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='the max number of epoch')
    parser.add_argument('--operation_num', type=int, default=5, help='the repeat operation of experiments')
    args = parser.parse_args()
    return args
def setup_logging(args, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    setlogger(os.path.join(save_dir, args.model_name + '.log'))
    logging.info("\n")
    time = datetime.strftime(datetime.now(), '%m-%d %H:%M:%S')
    logging.info('{}'.format(time))
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

def train_and_evaluate(args,operation, dataset):
    accuracy=[]
    for i  in range(args.operation_num):
        set_random_seed(42)
        target_loader_1, target_loader_2 = dataset.Loader([args.target_id], train=False)
        source_loader_list_1, source_loader_list_2 = dataset.Loader(args.source_id_list, train=True)
        source_list_string = "-".join(map(str, args.source_id_list))
        logging.info("Train_Source:{} Test_Target:{}".format(source_list_string, args.target_id))
        "Train"
        operation.setup(dataset.n_class)
        operation.train(i, source_loader_list_1, target_loader_1)
        "Test"
        acc= operation.test(i,source_loader_list_1,target_loader_1)
        accuracy.append(acc * 100)
    result_log(Indicators="Acc", target=args.target_id, source=args.source_list_string , results=accuracy)




if __name__ == '__main__':
    args = parse_args()
    Dataset = Fault_dataset(args)
    operation = train_test(args)
    setattr(args, 'num_class', Dataset.n_class)
    save_dir = os.path.join('./results/{}'.format(args.dataset_name))
    setup_logging(args, save_dir)
    train_and_evaluate(args,operation, Dataset)


