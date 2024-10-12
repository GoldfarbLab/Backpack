import sys
import numpy as np
import os
import yaml
from time import time
import utils_unispec
import csv
import torch
import wandb
from altimeter_dataset import AltimeterDataModule, filter_by_scan_range, match, norm_base_peak
from models import FlipyFlopy
from lightning_model import LitFlipyFlopy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
import lightning as L
from annotation import annotation
import matplotlib.pyplot as plt
plt.close('all')
torch.set_float32_matmul_precision('medium')

###############################################################################
############################### Configuration #################################
###############################################################################

with open(os.path.join(os.path.dirname(__file__), "../config/mods.yaml"), 'r') as stream:
    mod_config = yaml.safe_load(stream)
with open(os.path.join(os.path.dirname(__file__), "../config/data.yaml"), 'r') as stream:
    config = yaml.safe_load(stream)
D = utils_unispec.DicObj(config['ion_dictionary_path'], config['seq_len'], config['chlim'])


saved_model_path = os.path.join(config['base_path'], config['saved_model_path'])

# Configuration dictionary
if config['config'] is not None:
    # Load model config
    with open(config['config'], 'r') as stream:
        model_config = yaml.safe_load(stream)
else:
    channels = D.seq_channels
    model_config = {
        'in_ch': channels,
        'seq_len': D.seq_len,
        'out_dim': len(D.ion2index),
        **config['model_config']
    }

###############################################################################
############################ Weights and Biases ###############################
###############################################################################

wandb_logger = WandbLogger(project="Altimeter", config = config, log_model=False)


###############################################################################
################################## Model ######################################
###############################################################################

# Instantiate model
litmodel = LitFlipyFlopy(FlipyFlopy(**model_config), config)


# Load weights
if config['weights'] is not None:
    litmodel.model.load_state_dict(torch.load(config['weights']))

# TRANSFER LEARNING
if config['transfer'] is not None:
    litmodel.model.final = torch.nn.Sequential(torch.nn.Linear(512,D.dicsz), torch.nn.Sigmoid())
    for parm in litmodel.model.parameters(): parm.requires_grad=False
    for parm in litmodel.model.final.parameters(): parm.requires_grad=True
    
sys.stdout.write("Total model parameters: ")
litmodel.model.total_params()
    

###############################################################################
############################# Reproducability #################################
###############################################################################

model_folder_path = os.path.join(config['base_path'], config['saved_model_path'])
if not os.path.exists(model_folder_path): os.makedirs(model_folder_path)
with open(os.path.join(model_folder_path, "model_config.yaml"), "w") as file:
    yaml.dump(model_config, file)
with open(os.path.join(model_folder_path, "data.yaml"), "w") as file:
    yaml.dump(config, file)
with open(os.path.join(model_folder_path, "ion_dictionary.txt"), 'w') as file:
    file.write(open(config['ion_dictionary_path']).read())
    
def SpectralAngle(cs, eps=1e-5):
    cs = np.clip(cs, a_min=-(1-eps), a_max = 1-eps)
    return 1 - 2 * (np.arccos(cs) / np.pi)


###############################################################################
############################### Visualization #################################
###############################################################################

class MirrorPlotCallback(L.Callback):
    def on_validation_end(self, trainer, pl_module):
        val_dataset = dm.getAltimeterDataset("val")
        
        entry = val_dataset.get_target_plot(0)
        [sample, targ, mask, seq, mod, charge, nce, min_mz, max_mz, LOD, weight, moverz, annotated] = entry
        
        trainer.model.eval()
        with torch.no_grad():
            sample[0] = sample[0].unsqueeze(0)
            pred = trainer.model(sample)
            pred = pred.squeeze(0)
            pred, mzpred, ionspred = val_dataset.ConvertToPredictedSpectrum(pred.cpu().numpy(), seq, mod, charge)
            self.mirrorplot(entry, pred, mzpred, ionspred, pl_module, maxnorm=True, save=True)
    
    def mirrorplot(self, entry, pred, mzpred, ionspred, pl_module, maxnorm=True, save=True):

        [sample, targ, mask, seq, mod, charge, nce, min_mz, max_mz, LOD, weight, mz, annotated] = entry

        plt.close('all')

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
        
        
        
        targ, mz, annotated = filter_by_scan_range(mz, targ, min_mz, max_mz, annotated)
        targ_anno = targ[annotated]
        mz_anno = mz[annotated]
        pred, mz_pred, _ = filter_by_scan_range(mzpred, pred, min_mz, max_mz)



        targ_aligned, pred_aligned, mz_aligned = match(targ_anno, mz_anno, pred, mz_pred)
        
        targ_aligned = norm_base_peak(targ_aligned)
        pred_aligned = norm_base_peak(pred_aligned)
        
        pred_aligned[np.logical_and(pred_aligned <= LOD, targ_aligned == 0)] = 0
        
        cs = (pred_aligned*targ_aligned).sum() / max(np.linalg.norm(pred_aligned) * np.linalg.norm(targ_aligned), 1e-8)
        sa = 1 - 2 * (np.arccos(cs) / np.pi)
        
        charge = int(charge)
        nce = float(nce)
        annotated_percent = 100 * np.power(targ[annotated],2).sum() / np.power(targ,2).sum()
        ax.set_title(
            "Seq: %s(%d); Charge: +%d; NCE: %.2f; Mod: %s; Annotated: %.2f%%; SA=%.5f"%(
            seq, len(seq), charge, nce, mod, annotated_percent, sa)
        )
        
        pl_module.logger.experiment.log({"mirroplot": wandb.Image(plt)})
    




###############################################################################
########################## Training and testing ###############################
###############################################################################
stopping_criteria = EarlyStopping(monitor="val_SA_mean", mode="max", min_delta=0.00, patience=3)
checkpoint_callback = ModelCheckpoint(dirpath=saved_model_path, save_top_k=1, monitor="val_SA_mean", mode="max", every_n_epochs=1)
mirrorplot_callback = MirrorPlotCallback()

dm = AltimeterDataModule(config, D)
trainer = L.Trainer(default_root_dir=saved_model_path,
                    logger=wandb_logger,
                    callbacks=[stopping_criteria, checkpoint_callback, mirrorplot_callback],
                    strategy="ddp",
                    max_epochs=config['epochs'],
                    limit_train_batches=1000, 
                    limit_val_batches=1000
                    )


trainer.fit(litmodel, datamodule=dm)
trainer.test(datamodule=dm)

sys.exit()











def scoreDistPlot(losses, dataset, epoch=0):
    fig, ax = plt.subplots()
    ax.hist(losses, 30, histtype='bar', color='blue') #density=True,
    wandb_logger.log({"cs_dist_plot_" + dataset: wandb.Image(plt),
               "epoch": epoch})
    plt.close()

