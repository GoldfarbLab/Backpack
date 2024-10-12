import lightning as L
import torch as torch
import sys
from models import FlipyFlopy
import wandb
import matplotlib.pyplot as plt

# define the LightningModule
class LitFlipyFlopy(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        weighted_loss = self.step(batch, batch_idx)
        self.log("train_loss", weighted_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.config['batch_size'])  
        return weighted_loss
    
    def validation_step(self, batch, batch_idx):
        losses = self.test_step(batch, batch_idx)
        self.log("val_SA_mean", losses.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.config['test_batch_size']) 
        self.validation_step_outputs.append(losses)
        return losses
    
    def test_step(self, batch, batch_idx):
        losses = self.test_step(batch, batch_idx)
        self.log("test_SA_mean", losses.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.config['test_batch_size'])  
        return losses
    
    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs)
        self.scoreDistPlot(all_preds, "val")
        self.validation_step_outputs.clear()  # free memory
    
    def forward(self, x):
        return self.model(x)
    
    
    
    def step_helper(self, batch, batch_idx):
        samples = batch['samples']
        targ = batch['targ']
        mask = batch['mask']
        LODs = batch['LOD']
        weights = batch['weight']
        out = self.model(samples, test=False)
        return targ, mask, LODs, weights, out
    
    def step(self, batch, batch_idx):
        targ, mask, LODs, weights, out = self.step_helper(batch, batch_idx)
        weighted_loss = LossFunc(targ, out, mask, LODs, weights, root=self.config['root_int'])
        return weighted_loss
    
    def test_step(self, batch, batch_idx):
        targ, mask, LODs, weights, out = self.step_helper(batch, batch_idx)
        losses = -LossFunc(targ, out, mask, LODs, weights, root=self.config['root_int'], do_mean=False)
        return losses


    


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), eval(self.config['lr']))
        if self.config['restart'] is not None:
            # loading optimizer state requires it to be initialized with model GPU parms
            # how does this work with lightning?
            optimizer.load_state_dict(torch.load(self.config['restart']))
        return optimizer
    
    ###############################################################################
    ############################# Visualization ###################################
    ###############################################################################
    
    def mirrorplot(self, iloc=0, epoch=0, maxnorm=True, save=True, rank=-1, dataset="test"):
        plt.close('all')
        model.eval()
        model.to("cpu")
        
        if dataset=="test": 
            sample, info = L.input_from_str(telab[iloc:iloc+1])
            [targ, mz, annotated, mask] = L.target_plot(fposte[iloc], test_point)
        else: # assume validation
            sample, info = L.input_from_str(vallab[iloc:iloc+1])
            [targ, mz, annotated, mask] = L.target_plot(fposval[iloc], val_point)
        
        (seq, mod, charge, nce, min_mz, max_mz, LOD, iso2efficiency, weight) = info[0]
        
        with torch.no_grad():
            pred = model(sample)[0].squeeze().detach().numpy()
        pred, mzpred, ionspred = L.ConvertToPredictedSpectrum(pred, info[0], doIso = False)#(dataset=="test"))
        
        if maxnorm: pred /= pred.max()
        if maxnorm: targ /= targ.max()
        
        sort_pred = mzpred.argsort() # ion dictionary index to m/z ascending order
        sort_targ = mz.argsort()
        
        
        pred = pred[sort_pred]
        targ = targ[sort_targ]
        mz = mz[sort_targ]
        mzpred = mzpred[sort_pred]
        ionspred = ionspred[sort_pred]
        mask = mask[sort_targ]
        annotated = annotated[sort_targ]
        
        plt.close('all')
        fig,ax = plt.subplots()
        fig.set_figwidth(15)
        ax.set_xlabel("m/z")
        ax.set_ylabel("Intensity")
        
        if np.max(pred) > 0:
            max_mz_plot = max(np.max(mz), np.max(mzpred[pred > 0]))+10
        else:
            max_mz_plot = np.max(mz) + 10
        
        rect_lower = plt.Rectangle((0, -1), min_mz, 2, facecolor="#EEEEEE")
        rect_upper = plt.Rectangle((max_mz, -1), max_mz_plot, 2, facecolor="#EEEEEE")
        
        ax.add_patch(rect_lower)
        ax.add_patch(rect_upper)
        
        # plot annotated target peaks
        linestyles = ["solid" if m == 0 else (0, (1, 1)) for m in mask[annotated]]
        ax.vlines(mz[annotated], ymin=0, ymax=targ[annotated], linewidth=1, color='#111111', linestyle=linestyles)
        # plot unannotated target peaks
        ax.vlines(mz[annotated == False], ymin=0, ymax=targ[annotated == False], linewidth=1, color='#BBBBBB')
        
        
        
        # plot immonium
        plot_indices = np.logical_and(np.char.find(ionspred, "Int") != 0, np.char.find(ionspred, "I") == 0)
        colors = ["#ff7f00" if m >= LOD else "#BBBBBB" for m in pred[plot_indices]]
        ax.vlines(mzpred[plot_indices], ymin=-pred[plot_indices], ymax=0, linewidth=1, color=colors)
        
        # plot precursor
        plot_indices = np.logical_and(np.char.find(ionspred, "-") != 0, np.char.find(ionspred, "p") == 0)
        colors = ["#a65628" if m >= LOD else "#BBBBBB" for m in pred[plot_indices]]
        ax.vlines(mzpred[plot_indices], ymin=-pred[plot_indices], ymax=0, linewidth=1, color=colors)
        # plot precursor NLs
        plot_indices = np.logical_and(np.char.find(ionspred, "-") == 0, np.char.find(ionspred, "p") == 0)
        colors = ["#a65628" if m >= LOD else "#BBBBBB" for m in pred[plot_indices]]
        ax.vlines(mzpred[plot_indices], ymin=-pred[plot_indices], ymax=0, linewidth=1, color=colors, alpha=0.5)
        
        # plot b ions
        plot_indices = np.logical_and(np.char.find(ionspred, "-") != 0, np.char.find(ionspred, "b") == 0)
        colors = ["#e41a1c" if m >= LOD else "#BBBBBB" for m in pred[plot_indices]]
        ax.vlines(mzpred[plot_indices], ymin=-pred[plot_indices], ymax=0, linewidth=1, color=colors)
        # plot b NLs
        plot_indices = np.logical_and(np.char.find(ionspred, "-") == 0, np.char.find(ionspred, "b") == 0)
        colors = ["#e41a1c" if m >= LOD else "#BBBBBB" for m in pred[plot_indices]]
        ax.vlines(mzpred[plot_indices], ymin=-pred[plot_indices], ymax=0, linewidth=1, color=colors, alpha=0.5)
        
        # plot y ions
        plot_indices = np.logical_and(np.char.find(ionspred, "-") != 0, np.char.find(ionspred, "y") == 0)
        colors = ["#377eb8" if m >= LOD else "#BBBBBB" for m in pred[plot_indices]]
        ax.vlines(mzpred[plot_indices], ymin=-pred[plot_indices], ymax=0, linewidth=1, color=colors)
        # plot y NLs
        plot_indices = np.logical_and(np.char.find(ionspred, "-") == 0, np.char.find(ionspred, "y") == 0)
        colors = ["#377eb8" if m >= LOD else "#BBBBBB" for m in pred[plot_indices]]
        ax.vlines(mzpred[plot_indices], ymin=-pred[plot_indices], ymax=0, linewidth=1, color=colors, alpha=0.5)
        
        # plot other terminal ions
        ac_or = np.logical_or(np.char.find(ionspred, "a") == 0, np.char.find(ionspred, "c") == 0)
        xz_or = np.logical_or(np.char.find(ionspred, "x") == 0, np.char.find(ionspred, "z") == 0)
        other_term_or = np.logical_or(ac_or, xz_or)
        plot_indices = np.logical_and(np.char.find(ionspred, "-") != 0, other_term_or)
        colors = ["#f781bf" if m >= LOD else "#BBBBBB" for m in pred[plot_indices]]
        ax.vlines(mzpred[plot_indices], ymin=-pred[plot_indices], ymax=0, linewidth=1, color=colors)
        # plot other terminal NLs
        plot_indices = np.logical_and(np.char.find(ionspred, "-") == 0, other_term_or)
        colors = ["#f781bf" if m >= LOD else "#BBBBBB" for m in pred[plot_indices]]
        ax.vlines(mzpred[plot_indices], ymin=-pred[plot_indices], ymax=0, linewidth=1, color=colors, alpha=0.5)
        
        # plot internal ions
        plot_indices = np.logical_and(np.char.find(ionspred, "-") != 0, np.char.find(ionspred, "Int") == 0)
        colors = ["#984ea3" if m >= LOD else "#BBBBBB" for m in pred[plot_indices]]
        ax.vlines(mzpred[plot_indices], ymin=-pred[plot_indices], ymax=0, linewidth=1, color=colors)
        # plot internal NLs
        plot_indices = np.logical_and(np.char.find(ionspred, "-") == 0, np.char.find(ionspred, "Int") == 0)
        colors = ["#984ea3" if m >= LOD else "#BBBBBB" for m in pred[plot_indices]]
        ax.vlines(mzpred[plot_indices], ymin=-pred[plot_indices], ymax=0, linewidth=1, color=colors, alpha=0.5)
        
        
        ax.set_xlim([0, ax.get_xlim()[1]])
        ax.set_ylim([-1.1,1.1])
        ax.set_xlim([0, max_mz_plot])
        ax.set_xticks(np.arange(0,ax.get_xlim()[1],200))
        ax.set_xticks(np.arange(0,ax.get_xlim()[1],50), minor=True)
        
        
        
        targ, mz, annotated = L.filter_by_scan_range(mz, targ, min_mz, max_mz, annotated)
        targ_anno = targ[annotated]
        mz_anno = mz[annotated]
        pred, mz_pred, _ = L.filter_by_scan_range(mzpred, pred, min_mz, max_mz)



        targ_aligned, pred_aligned, mz_aligned = L.match(targ_anno, mz_anno, pred, mz_pred)
        
        targ_aligned = L.norm_base_peak(targ_aligned)
        pred_aligned = L.norm_base_peak(pred_aligned)
        
        pred_aligned[np.logical_and(pred_aligned <= LOD, targ_aligned == 0)] = 0
        
        #targ_aligned = np.sqrt(targ_aligned)
        #pred_aligned = np.sqrt(pred_aligned)
        
        cs = (pred_aligned*targ_aligned).sum() / max(np.linalg.norm(pred_aligned) * np.linalg.norm(targ_aligned), 1e-8)
        #sys.stdout.write("%.3f"%(cs))
        #sys.stdout.write("\n")
        sa = 1 - 2 * (np.arccos(cs) / np.pi)
        mae  = abs(pred_aligned[pred_aligned>0.05]-targ_aligned[pred_aligned>0.05]).mean()
        norm_pred = L.norm_sum_one(pred_aligned)
        norm_targ = L.norm_sum_one(targ_aligned)
        scribe = -np.log(np.sum(np.power(norm_pred - norm_targ, 2)))
        
        charge = int(charge)
        nce = float(nce)
        iso = str(min(iso2efficiency))+"-"+str(max(iso2efficiency))
        annotated_percent = 100 * np.power(targ[annotated],2).sum() / np.power(targ,2).sum()
        ax.set_title(
            "Seq: %s(%d); Charge: +%d; NCE: %.2f; Mod: %s; Iso: %s; Annotated: %.1f%%; SA=%.3f; MAE: %.4f; Scribe: %.4f"%(
            seq, len(seq), charge, nce, mod, iso, annotated_percent, sa, mae, scribe)
        )
        
        if rank == -1: rank = iloc

        
            
        wandb.log({"mirroplot_%s_%d"%(dataset,rank): wandb.Image(plt),
                "epoch": epoch})
    
    def scoreDistPlot(self, losses, dataset, epoch=0):
        plt.close('all')
        fig, ax = plt.subplots()
        ax.hist(losses.cpu(), 30, histtype='bar', color='blue') #density=True,
        self.logger.experiment.log({"cs_dist_plot_" + dataset: wandb.Image(plt)})
        plt.close()
        
    
###############################################################################
############################# Loss function ###################################
###############################################################################
CS = torch.nn.CosineSimilarity(dim=-1)

def LossFunc(targ, pred, mask, LOD, weights, root, doFullMask=True, epsilon=1e-5, do_weights=True, do_mean=True): 
    #pred /= torch.max(pred, dim=1, keepdim=True)[0]
    pred /= torch.sum(pred, dim=1, keepdim=True)[0]
    
    #weights = torch.unsqueeze(weights, 1)
    #pred = torch.unsqueeze(pred, 1)
    targ = torch.squeeze(targ, 1)
    mask = torch.squeeze(mask, 1)
    
    targ, pred = apply_mask(targ, pred, mask, LOD, doFullMask)
    
    targ = root_intensity(targ, root=root) if root is not None else targ
    pred = root_intensity(pred, root=root) if root is not None else pred
    
    cs = CS(targ, pred)
    cs = torch.clamp(cs, min=-(1-epsilon), max=(1-epsilon))
    sa = -(1 - 2 * (torch.arccos(cs) / torch.pi))
    if torch.any(torch.isnan(sa)):
        print("nan unweighted SA")
        sys.exit()
        
    #weighted = sa * weights
    #weighted = weighted.sum() / weights.sum()
    #return -weighted
    
    if do_mean:
        sa = sa.mean()
    
    return sa

def apply_mask(targ, pred, mask, LOD, doFullMask=True):
    #LOD = torch.reshape(LOD, (LOD.shape[0], 1, 1)).expand_as(targ)
    LOD = torch.reshape(LOD, (LOD.shape[0], 1)).expand_as(targ)
    
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



