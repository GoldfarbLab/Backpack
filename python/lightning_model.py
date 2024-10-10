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
        samples, targ, mask, info = batch
        out,_,_ = self.model(samples, test=False)
        weighted_loss = LossFunc(targ, out, mask, info, root=self.config['root_int'])
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

def LossFunc(targ, pred, mask, info, root, doFullMask=True, epsilon=1e-5):
    LOD = [s[6] for s in info]
    LOD = torch.FloatTensor(LOD)
    
    weights = [s[8] for s in info]
    weights = torch.FloatTensor(weights)

    #pred /= torch.max(pred, dim=1, keepdim=True)[0]
    pred /= torch.sum(pred, dim=1, keepdim=True)[0]
    targ, pred = L.apply_mask(targ, pred, mask, LOD, doFullMask)
    
    targ = L.root_intensity(targ, root=root) if root is not None else targ
    pred = L.root_intensity(pred, root=root) if root is not None else pred
    
    cs = torch.nn.CosineSimilarity(targ, pred, dim=-1)
    cs = torch.clamp(cs, min=-(1-epsilon), max=(1-epsilon))
    sa = 1 - 2 * (torch.arccos(cs) / torch.pi)
    if torch.any(torch.isnan(sa)):
        print("nan unweighted SA")
        sys.exit()
    weighted = sa * weights

    return -weighted