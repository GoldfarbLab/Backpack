from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
import numpy as np
import torch as torch
import re as re
import os

class AltimeterDataModule(LightningDataModule):
    def __init__(self, config, D):
        super().__init__()
        self.config = config
        self.D = D
        
    def getAltimeterDataset(self, dataset):
        pos_path = os.path.join(self.config['base_path'], self.config['position_path'], "fpos" + dataset + ".txt")
        data_path = os.path.join(self.config['base_path'], self.config['dataset_path'], dataset + ".txt")
        lab_path = os.path.join(self.config['base_path'], self.config['label_path'], dataset + "_labels.txt")
        return AltimeterDataset(pos_path, lab_path, data_path, self.D, num_workers=self.config['num_workers'])

    def setup(self, stage: str):
        if stage == "fit":
            self.dataset_train = self.getAltimeterDataset("train")
            self.dataset_val = self.getAltimeterDataset("val")
        elif stage == "test":
            self.dataset_test = self.getAltimeterDataset("test")

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.config['test_batch_size'], shuffle=False, num_workers=self.config['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.config['test_batch_size'], shuffle=False, num_workers=self.config['num_workers'])
    
    def predict_dataloader(self):
        pass
        #return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass

class AltimeterDataset(Dataset):
    def __init__(self, pos_path, label_path, data_path, dobj, num_workers=1, transform=None):
        
        self.positions = np.loadtxt(pos_path).astype(int)
        self.labels = np.array([line.strip() for line in open(label_path,'r')])
        self.worked_id2fdata = dict()
        for worker_id in range(num_workers):
             self.worked_id2fdata[worker_id] = open(data_path, "r")
        
        self.D = dobj
        self.channels = dobj.seq_channels
        self.transform = transform
        
        

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        samples, info = self.input_from_str(self.labels[idx])
        seq, mod, charge, nce, min_mz, max_mz, LOD, _, weight = info
        targ, _, mask = self.target(self.worked_id2fdata[worker_info.id], self.positions[idx], return_mz=False)
        
        sample = {'samples': samples, 'targ': targ, 'mask': mask, 'seq': seq, 'mod': mod, 'charge': charge, 'nce': nce, 'min_mz': min_mz, 'max_mz': max_mz, 'LOD': LOD, 'weight': weight}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def input_from_str(self, strings):
        [seq,other] = strings.split('/')
        osplit = other.split("_")
        [charge, mod, nce, scan_range, LOD, iso_efficiency, weight] = osplit
        charge = int(charge)
        nce = float(nce[3:])
        min_mz = float(scan_range.split("-")[0])
        max_mz = float(scan_range.split("-")[1])
        LOD = float(LOD)
        weight = float(weight)
        iso2efficiency = dict()
        for iso_entry in iso_efficiency.split(","):
            if iso_entry == "": iso2efficiency[0] = 1
            else:
                isotope = int(iso_entry.split(")")[-1])
                efficiency = float(iso_entry.split(")")[0][1:])
                iso2efficiency[isotope] = efficiency
            
        info = (seq, mod, charge, nce, min_mz, max_mz, LOD, iso2efficiency, weight)
        out = self.inptsr(info)
        return out, info
    
    def target(self, fp, fstart, mint=0, return_mz=False):
        """
        Create target, from streamlined dataset, to train model on.
        
        :param fstart: array of file positions for spectra to be predicted.
        :param fp: filepointer to streamlined dataset.
        :param mint: minimum intensity to include in target spectrum.
        :param return_mz: whether to return the corresponding m/z values for
                          fragment ions.
        
        :return target: pytorch array of intensities for all ions in output
                        output space.
        :return moverz: pytorch array of m/z values corresponding to ions
                        present in target array. All zeros if return_mz=False.
        """
        
        target = torch.full(
            (1, self.D.dicsz), mint, dtype=torch.float32
        )
        moverz = (torch.zeros((1, self.D.dicsz), dtype=torch.float32) if return_mz else 0)
        masks = torch.full((1, self.D.dicsz), mint, dtype=torch.float32)
        
        # Fill the output
        fp.seek(fstart)
        nmpks = int(fp.readline().split()[1].split("|")[-1])
        for pk in range(nmpks):
            line = fp.readline()
            [d,I,mz,intensity,mask] = line.split()
            I = int(I)
            if I == -1: continue
            target[0,I] = float(intensity)
            masks[0,I] = int(mask)
            if return_mz: moverz[0,I] = float(mz)

        return target, moverz, masks
    
    def inptsr(self, info):
        """
        Create input(s) for 1 peptide

        Parameters
        ----------
        info : tuple of (seq,mod,charge,nce)

        Returns
        -------
        out : List of a) tensor to model, b) charge float and/or c) ce float.
              Only outputs a) if not self.embed

        """
        (seq, mod, charge, nce, _, _, _, _, _) = info
        output = torch.zeros((self.channels, self.D.seq_len), dtype=torch.float32)
        
        # Sequence
        assert len(seq) <= self.D.seq_len, "Exceeded maximum peptide length."
        output[:len(self.D.dic),:len(seq)] = torch.nn.functional.one_hot(
            torch.tensor([self.D.dic[o] for o in seq], dtype=torch.long),
            len(self.D.dic)
        ).T
        output[len(self.D.dic)-1, len(seq):] = 1.
        # PTMs
        Mstart = mod.find('(') if mod!='0' else 1
        modamt = int(mod[0:Mstart])
        output[len(self.D.dic)] = 1.
        if modamt>0:
            hold = [re.sub('[()]', '', n) for n in mod[Mstart:].split(")(")]
            for n in hold:
                [pos,aa,modtyp] = n.split(',')
                output[self.D.mdic[modtyp], int(pos)] = 1.
                output[len(self.D.dic), int(pos)] = 0.
        
        out = [output, float(charge), float(nce)]

        return out