import logging
import os
import time
import torch
from torch import nn
from torch import optim
import models

class train_test(object):
    def __init__(self, args):
        self.args = args
    def setup(self,n_class):
        args = self.args
        self.method=args.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.method in  ['M4']:
            self.model_name='SSDG'
        self.model =getattr(models,self.model_name)(in_channel=1,num_classes=n_class,lr=args.lr,set=self.method)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=0.0001)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        return
    def train(self, op_num,TR_dataloader,TS_dataloader_S):
        best_epoch=0
        args = self.args
        best_acc = 0.0
        time_all=0.0
        if self.model_name == 'SSDG':
            for epoch in range(args.epoch):
                epoch_start = time.time()
                loss=self.model(TR_dataloader,epoch_it=epoch)
                logging.info('Num-{}, Epoch: {}, Loss: {:.2f},Time {:.4f} sec'.format(op_num, epoch, loss,time.time() - epoch_start))
                time_all+= time.time() - epoch_start

                self.model.eval()
                total_correct = 0
                total_samples = 0
                with torch.no_grad():
                    for batch_idx, (inputs, labels,domain, _) in enumerate(TS_dataloader_S):
                        labels = labels.to(self.device)
                        logits = self.model.model_inference(inputs)
                        pred = logits.argmax(dim=1)
                        total_correct += (pred == labels).sum().item()
                        total_samples += labels.size(0)

                    epoch_acc = (total_correct / total_samples)*100
                    logging.info("Target Domain Accuracy: {:.2f}".format(epoch_acc))
                    if epoch_acc >= best_acc:
                        best_acc = epoch_acc
                        best_epoch = epoch
                        save_dir = os.path.join('./trained_models/{}/{}'.format(args.dataset_name, args.model_name))
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        torch.save(self.model.state_dict(), os.path.join('{}/{}.pth'.format(save_dir, 'operation_' + str(op_num))))


            logging.info("Training time {:.2f}, Best_epoch {}".format(time_all, best_epoch))

        return

    def test(self, op_num, TR_dataloader, TS_dataloader):
        args = self.args
        save_dir = os.path.join(f'./trained_models/{args.dataset_name}/{args.model_name}')
        model_path = os.path.join(save_dir, f'operation_{op_num}.pth')

        # 加载模型参数
        self.model.load_state_dict(torch.load(model_path), strict=False)
        self.model.eval()

        # 初始化变量
        total_acc, total_samples = 0.0, 0.0
        test_start_time = time.time()
        with torch.no_grad():
            for inputs, labels, domain, _ in TS_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if args.model_name == 'M4':
                    logits = self.model.model_inference(inputs)

                predictions = logits.argmax(dim=1)
                correct_predictions = (predictions == labels).sum().item()
                batch_size = labels.size(0)
                total_acc += correct_predictions
                total_samples += batch_size


        avg_acc = total_acc / total_samples
        test_duration = time.time() - test_start_time

        logging.info(
            f'Operation_{op_num}, Acc: {avg_acc:.4f}, Time: {test_duration:.4f} sec')

        return avg_acc




