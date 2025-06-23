import torch.utils.data as Data
from torch.utils.data import Dataset
import torch
import scipy.io as scio
from scipy.fftpack import fft
import numpy as np

data_pth={
    "GearBox.BJUT":'.\data\GearBox.BJUT\GearData',
          }
classes_map = {
    'GearBox.BJUT': 5,
}
domain_map = {
    'GearBox.BJUT': [1200,1800,2400,3000],
}
def get_domain_file(name):
    if name not in data_pth:
        raise ValueError('Name of datasetpu unknown %s' %name)
    return data_pth[name]
def get_domain_task(name):
    if name not in domain_map.keys():
        raise ValueError('Name of datasetpu unknown %s' %name)
    return domain_map[name]

class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor, target_tensor,train_y_domain_label):
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.target_domain = train_y_domain_label

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index],self.target_domain[index],index
class Fault_dataset(Dataset):
    def __init__(self,args):
        self.args = args
        self.data_pth =get_domain_file(self.args.dataset_name)
        self.task=get_domain_task(self.args.dataset_name)
        self.n_class=classes_map[self.args.dataset_name]-len(self.args.miss_class)

    def load_data(self,path, temp, sum_class,data_ratio, mis_class, FFT=True, normalize_type="0-1"):
        data_temp = scio.loadmat(path)
        data = data_temp.get(temp)
        self.n_class = int(sum_class) - len(mis_class)
        for i_c in mis_class:
            mis_class_id = np.argwhere(data[:, -1] == i_c)
            data = np.delete(data, mis_class_id, axis=0)
        class_sample, _ = data.shape
        sample_ratio=int(class_sample * data_ratio)
        train_x= data[:sample_ratio, :-1]
        train_y = data[:sample_ratio, -1]
        test_x= data[sample_ratio:, :-1]
        test_y = data[sample_ratio:, -1]
        if FFT:
            train_x = np.abs(fft(train_x, axis=1))[:, :512]
            test_x =np.abs(fft(test_x , axis=1))[:, :512]
        train_x = self.Normalize(train_x, normalize_type)
        test_x = self.Normalize(test_x, normalize_type)
        train_x = torch.FloatTensor(train_x).unsqueeze(1)
        train_y = torch.LongTensor(train_y)
        test_x = torch.FloatTensor(test_x).unsqueeze(1)
        test_y = torch.LongTensor(test_y)
        return train_x, train_y, test_x, test_y

    def Normalize(self,data, type):
        seq = data
        if type == "0-1":
            Zmax, Zmin = seq.max(axis=1), seq.min(axis=1)
            seq = (seq - Zmin.reshape(-1, 1)) / (Zmax.reshape(-1, 1) - Zmin.reshape(-1, 1))
        elif type == "1-1":
            seq = 2 * (seq - seq.min()) / (seq.max() - seq.min()) + -1
        elif type == "mean-std":
            mean = np.mean(seq, axis=1, keepdims=True)
            std = np.std(seq, axis=1, keepdims=True)
            std[std == 0] = 1
            seq = (seq - mean) / std
        else:
            seq=seq

        return seq


    def Loader(self,data_list_name=[],train=False):
        train_loader_x = []
        test_loader_x = []
        for domain_id,domain in enumerate(data_list_name):
            sum_class = classes_map[self.args.dataset_name]
            root=data_pth[self.args.dataset_name]+"_"+str(domain)+"_"+str(sum_class)+".mat"
            temp= root.split('\\')[-1].split('.')[0]
            train_x, train_y, test_x, test_y= self.load_data(root,temp, sum_class,self.args.data_ratio, self.args.miss_class, self.args.FFT, self.args.normalize_type)
            train_domain=torch.full_like(train_y,domain_id)
            train_dataset = CustomTensorDataset(train_x, train_y,train_domain)
            test_domain=torch.full_like(test_y,domain_id)
            test_dataset = CustomTensorDataset(test_x, test_y,test_domain)
            if self.args.dataset_name=="GearBox.BJUT":
                batch_size=800
            if train:
                train_loader_x.append(train_dataset)
                test_loader_x.append(test_dataset)
            else:
                train_loader_x = Data.DataLoader(train_dataset, batch_size, drop_last=True)
                test_loader_x = Data.DataLoader(test_dataset, batch_size, drop_last=True)
        return train_loader_x, test_loader_x




