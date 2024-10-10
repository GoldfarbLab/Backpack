from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch as torch

class AltimeterDataset(Dataset):
    def __init__(self, pos_path, label_path, data_path, transform=None):
        
        self.positions = np.loadtxt(pos_path).astype(int)
        self.labels = np.array([line.strip() for line in open(label_path,'r')])
        self.fdata = open(data_path, "r")
        self.transform = transform

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample