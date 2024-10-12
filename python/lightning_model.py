import lightning as L
import torch as torch
import sys
from models import FlipyFlopy


# define the LightningModule
class LitFlipyFlopy(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

    def training_step(self, batch, batch_idx):
        weighted_loss = self.step(batch, batch_idx)
        self.log("train_loss", weighted_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.config['batch_size'])  
        return weighted_loss
    
    def validation_step(self, batch, batch_idx):
        weighted_loss = -self.step(batch, batch_idx)
        self.log("val_SA_mean", weighted_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.config['test_batch_size'])  
        return weighted_loss
    
    def test_step(self, batch, batch_idx):
        weighted_loss = -self.step(batch, batch_idx)
        self.log("test_SA_mean", weighted_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.config['test_batch_size'])  
        return weighted_loss
        
    def step(self, batch, batch_idx):
        samples = batch['samples']
        targ = batch['targ']
        mask = batch['mask']
        LODs = batch['LOD']
        weights = batch['weight']
        out,_,_ = self.model(samples, test=False)
        weighted_loss = LossFunc(targ, out, mask, LODs, weights, root=self.config['root_int'])
        
        return weighted_loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), eval(self.config['lr']))
        if self.config['restart'] is not None:
            # loading optimizer state requires it to be initialized with model GPU parms
            # how does this work with lightning?
            optimizer.load_state_dict(torch.load(self.config['restart']))
        return optimizer
    
    
###############################################################################
############################# Loss function ###################################
###############################################################################
CS = torch.nn.CosineSimilarity(dim=-1)

def LossFunc(targ, pred, mask, LOD, weights, root, doFullMask=True, epsilon=1e-5): 
    #pred /= torch.max(pred, dim=1, keepdim=True)[0]
    pred /= torch.sum(pred, dim=1, keepdim=True)[0]
    
    weights = torch.unsqueeze(weights, 1)
    pred = torch.unsqueeze(pred, 1)
    targ, pred = apply_mask(targ, pred, mask, LOD, doFullMask)
    
    targ = root_intensity(targ, root=root) if root is not None else targ
    pred = root_intensity(pred, root=root) if root is not None else pred
    
    cs = CS(targ, pred)
    cs = torch.clamp(cs, min=-(1-epsilon), max=(1-epsilon))
    sa = 1 - 2 * (torch.arccos(cs) / torch.pi)
    if torch.any(torch.isnan(sa)):
        print("nan unweighted SA")
        sys.exit()
        
    #weighted = sa * weights
    #weighted = weighted.sum() / weights.sum()
    #return -weighted
    return -sa.mean()

def apply_mask(targ, pred, mask, LOD, doFullMask=True):
    LOD = torch.reshape(LOD, (LOD.shape[0], 1, 1)).expand_as(targ)
    # mask below limit of detection
    pred = torch.where(torch.logical_and(targ==0, pred<=LOD), 0.0, pred)  
    if doFullMask:
        pred = torch.where(torch.logical_and(targ==0, pred>LOD), pred-LOD, pred)

    
    # mask 1 - outside of scan range. Can have any intensity without penalty
    pred = torch.where(mask==1, 0.0, pred)
    targ = torch.where(mask==1, 0.0, targ)
    
    # mask 2-5 - bad isotope dist, below purity, high m/z error, ambiguous annotation. Can have any intensity up to the target
    if doFullMask:
        pred = torch.where(torch.logical_and(mask>1, pred < targ), 0.0, pred)
        pred = torch.where(torch.logical_and(mask>1, pred > targ), pred-targ, pred)
        targ = torch.where(mask>1, 0.0, targ)
        
        #pred = torch.where(torch.logical_and(torch.logical_and(mask>1, mask !=3), pred < targ), 0.0, pred)
        #pred = torch.where(torch.logical_and(torch.logical_and(mask>1, mask !=3), pred > targ), pred-targ, pred)
        #targ = torch.where(torch.logical_and(mask>1, mask !=3), 0.0, targ)
        
    return targ, pred
    
def root_intensity(ints, root=2):
    if root==2:
        ints[ints>0] = torch.sqrt(ints[ints>0]) # faster than **(1/2)
    else:
        ints[ints>0] = ints[ints>0]**(1/root)
    return ints