import sys
import numpy as np
import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
import yaml
from time import time
import utils_unispec
import csv
import torch
import wandb
from models import FlipyFlopy
from annotation import annotation
import matplotlib.pyplot as plt
plt.close('all')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
wandb.require("core")


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
    channels = D.seq_channels if config['model_config']['CEembed'] else D.channels
    model_config = {
        'in_ch': channels,
        'seq_len': D.seq_len,
        'out_dim': len(D.ion2index),
        **config['model_config']
    }

###############################################################################
############################ Weights and Biases ###############################
###############################################################################

wandb.login()

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Altimeter",
    # track hyperparameters and run metadata
    config = config
)

# define a metrics
wandb.define_metric("val.loss", summary="max")
wandb.define_metric("test.loss", summary="max")


###############################################################################
################################ Dataset ######################################
###############################################################################

from utils_unispec import LoadObj
L = LoadObj(D, embed=model_config['CEembed'])

# Training
fpostr = np.loadtxt(os.path.join(config['base_path'], config['position_path'], "fpostrain.txt")).astype(int)
ftr = open(os.path.join(config['base_path'], config['dataset_path'], "train.txt"), "r")
trlab = np.array([line.strip() for line in open(os.path.join(config['base_path'], config['label_path'], "train_labels.txt"),'r')])

# validation
fposval = np.loadtxt(os.path.join(config['base_path'], config['position_path'], "fposval.txt")).astype(int)
val_point = open(os.path.join(config['base_path'], config['dataset_path'], "val.txt"), "r")
vallab = np.array([line.strip() for line in open(os.path.join(config['base_path'], config['label_path'], "val_labels.txt"),'r')])

# testing
fposte = np.loadtxt(os.path.join(config['base_path'], config['position_path'], "fpostest.txt")).astype(int)
test_point = open(os.path.join(config['base_path'], config['dataset_path'], "test.txt"), "r")
telab = np.array([line.strip() for line in open(os.path.join(config['base_path'], config['label_path'], "test_labels.txt"),'r')])

# find long sequence for mirrorplot
Lens = []
Lens_peaks = []
for pos in fposte:
    test_point.seek(pos) 
    line = test_point.readline()
    Lens_peaks.append(len(line.split()[1].split('|')[0]) + int(line.split()[1].split('|')[-1]))
    Lens.append(len(line.split()[1].split('|')[0]))
MPIND = 365 #np.argmax(Lens) #1601 #np.argmax(Lens)
MPIND2 = np.argmax(Lens_peaks)

###############################################################################
################################## Model ######################################
###############################################################################

# Instantiate model
model = FlipyFlopy(**model_config, device=device)
arrdims = len(model(L.input_from_str(trlab[0:1])[0], test=True)[1][0])
model.to(device)


# Load weights
if config['weights'] is not None:
    model.load_state_dict(torch.load(config['weights']))

# TRANSFER LEARNING
if config['transfer'] is not None:
    model.final = torch.nn.Sequential(torch.nn.Linear(512,D.dicsz), torch.nn.Sigmoid())
    for parm in model.parameters(): parm.requires_grad=False
    for parm in model.final.parameters(): parm.requires_grad=True
    
sys.stdout.write("Total model parameters: ")
model.total_params()
    
# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs for training.")
    model = torch.nn.DataParallel(model)  # Wrap the model with DataParallel

# Optimizer
opt = torch.optim.Adam(model.parameters(), eval(config['lr']))
if config['restart'] is not None:
    # loading optimizer state requires it to be initialized with model GPU parms
    opt.load_state_dict(torch.load(config['restart'], map_location=device))
    
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
    
###############################################################################
############################# Loss function ###################################
###############################################################################

CS = torch.nn.CosineSimilarity(dim=-1)
def LossFunc(targ, pred, mask, info, root=config['root_int'], doFullMask=True, epsilon=1e-5):
    LOD = [s[6] for s in info]
    LOD = torch.FloatTensor(LOD).to(device)
    
    weights = [s[8] for s in info]
    weights = torch.FloatTensor(weights).to(device)

    #pred /= torch.max(pred, dim=1, keepdim=True)[0]
    pred /= torch.sum(pred, dim=1, keepdim=True)[0]
    targ, pred = L.apply_mask(targ, pred, mask, LOD, doFullMask)
    
    targ = L.root_intensity(targ, root=root) if root is not None else targ
    pred = L.root_intensity(pred, root=root) if root is not None else pred
    
    cs = CS(targ, pred)
    cs = torch.clamp(cs, min=-(1-epsilon), max=(1-epsilon))
    sa = 1 - 2 * (torch.arccos(cs) / torch.pi)
    if torch.any(torch.isnan(sa)):
        print("nan unweighted SA")
        sys.exit()
    weighted = sa * weights
    weighted = weighted.sum() / weights.sum()

    return -weighted, sa #cs

def ScribeLoss(targ, pred, root=config['root_int']):
    targ, pred = L.apply_mask(targ, pred)
    targ = L.root_intensity(targ, root=root) if root is not None else targ
    pred = L.root_intensity(pred, root=root) if root is not None else pred
    scribe = -torch.log(torch.sum(torch.pow(targ - pred, 2)))
    return -scribe

def SpectralAngle(cs):
    return 1 - 2 * (np.arccos(cs) / np.pi)




###############################################################################
########################## Training and testing ###############################
###############################################################################

def train_step(samples, targ, mask, info):
    
    samplesgpu = [m.to(device) for m in samples]
    targgpu = targ.to(device)
    maskgpu = mask.to(device)
    
    model.to(device)
    
    model.train()
    model.zero_grad()
    out,_,_ = model(samplesgpu, test=False)
    
    weighted_loss, losses = LossFunc(targgpu, out, maskgpu, info, root=config['root_int'])
    #loss = loss.mean()
    weighted_loss.backward()
    opt.step()

    return weighted_loss, losses

def Testing(labels, pos, pointer, batch_size, outfile=None):
    with torch.no_grad():
        model.eval()
        tot = len(labels)
        steps = (tot//batch_size) if tot%batch_size==0 else (tot//batch_size)+1
        model.to(device)
        Loss = 0
        losses = []
        arr = torch.zeros(config['model_config']['blocks'], arrdims)
        for m in range(steps):
            begin = m*batch_size
            end = (m+1)*batch_size
            
            if m % 1000 == 0: print("Testing", "Step:", m, "/", steps)
            # Test set
            targ, _, mask = L.target(pos[begin:end], fp=pointer, return_mz=False)
            samples, info = L.input_from_str(labels[begin:end])
            samplesgpu = [m.to(device) for m in samples]
            maskgpu = mask.to(device)
            
            out,out2,FMs = model(samplesgpu)

            weighted_loss, ind_losses = LossFunc(targ.to(device), out, maskgpu, info, doFullMask=False)
            losses_list = (-ind_losses).tolist()
            losses.extend(losses_list)
            #losses.append(loss)
            Loss += weighted_loss #.sum()
            arr += torch.tensor([[n for n in m] for m in out2])
            
            samples_info = [n for n in 
                          L.input_from_str(labels[begin:end])[1]
            ]
            
            if outfile:
                #targ.to('cpu').detach()
                # loop through labels
                for i in range(len(samples_info)):
                    seq, mod, charge, nce, min_mz, max_mz, LOD, iso2efficiency, weight = samples_info[i]
                    targ_int = targ[i].numpy()
                    pred_int = out[i].cpu().numpy()
                    pred_int /= np.max(pred_int)
                    for j in range(D.dicsz):
                        # parse ion name
                        annot = annotation.from_entry(D.index2ion[j], charge)
                        # write info, loss, individual ions
                        outfile.writerow([seq, str(len(seq)), mod, str(charge), "{:.2f}".format(nce), "{:.3f}".format(losses_list[i]), 
                                          # full name, type, NL string, charge
                                          annot.getName(), annot.getType(), annot.getNLString(), annot.z, 
                                          "{:.3f}".format(targ_int[j]), "{:.3f}".format(pred_int[j])])
                    
    model.to('cpu')
    Loss = (Loss/steps).to('cpu').detach().numpy()
    return Loss, arr.detach().numpy() / steps, np.array(losses)

def Testing_np(labels, pos, pointer, batch_size):
    with torch.no_grad():
        model.eval()
        tot = len(labels)
        steps = (tot//batch_size) if tot%batch_size==0 else (tot//batch_size)+1
        model.to(device)
        Loss = 0
        losses = []
        arr = torch.zeros(config['model_config']['blocks'], arrdims)
        for m in range(steps):
            begin = m*batch_size
            end = (m+1)*batch_size
            # Test set
            #targ, mz = L.target(pos[begin:end], fp=pointer, return_mz=True)
            [targ, mz, annotated, mask] = L.target_plot(pos[begin], pointer)
            samples, info = L.input_from_str(labels[begin:end])
            (seq, mod, charge, nce, min_mz, max_mz, LOD, iso2efficiency, weight) = info[0]
            samplesgpu = [n.to(device) for n in samples]
            out, out2, FMs = model(samplesgpu)
            
            pred, mzpred, ionspred = L.ConvertToPredictedSpectrum(out.cpu().numpy()[0], info[0])
            
            sort_pred = mzpred.argsort() # ion dictionary index to m/z ascending order
            sort_targ = mz.argsort()
            
            pred = pred[sort_pred]
            targ = targ[sort_targ]
            mz = mz[sort_targ]
            mzpred = mzpred[sort_pred]
            ionspred = ionspred[sort_pred]
            mask = mask[sort_targ]
            annotated = annotated[sort_targ]
            
            targ, mz, annotated = L.filter_by_scan_range(mz, targ, min_mz, max_mz, annotated)
            targ_anno = targ[annotated]
            mz_anno = mz[annotated]
            pred, mz_pred, _ = L.filter_by_scan_range(mzpred, pred, min_mz, max_mz)
            
            if pred.size > 0 or mz_pred.size > 0:
                targ_aligned, pred_aligned, _ = L.match(targ_anno, mz_anno, pred, mz_pred)
                
                targ_aligned = L.norm_base_peak(targ_aligned)
                pred_aligned = L.norm_base_peak(pred_aligned)

                pred_aligned[np.logical_and(pred_aligned <= LOD, targ_aligned == 0)] = 0
                
                targ_aligned = np.sqrt(targ_aligned)
                pred_aligned = np.sqrt(pred_aligned)
                
                cs = (pred_aligned*targ_aligned).sum() / max(np.linalg.norm(pred_aligned) * np.linalg.norm(targ_aligned), 1e-8)
                sys.stdout.write("test %.3f"%(cs) + "\n")
                sa = 1 - 2 * (np.arccos(cs) / np.pi)
                
                #loss = LossFunc(targ.to(device), out)
                Loss += -sa#loss.sum()
                losses_list = (sa).tolist()
                losses.extend([losses_list])
            arr += torch.tensor([[n for n in m] for m in out2])
    model.to('cpu')
    Loss = (Loss/tot)#.to('cpu').detach().numpy()
    return Loss, arr.detach().numpy() / steps, losses




def train(epochs,
          batch_size=100,
          lr_decay_start = 1e10,
          lr_decay_rate = 0.9,
          shuffle=True, 
          svwts=False):
    
    print("Starting training for %d epochs"%epochs)
    tot = len(trlab)
    steps = np.minimum(
        config['steps'] if config['steps'] is not None else 1e10,
        tot//batch_size if tot%batch_size==0 else tot//batch_size + 1
    )
    #steps = tot//batch_size if tot%batch_size==0 else tot//batch_size + 1
    
    currbest = 0
    test_loss, _ = 0,0
    val_loss, varr = 0,0
    
    # Plot initial mirrors to see the ion dictionary
    mirrorplot(MPIND)
    mirrorplot(MPIND2)
    
    # Training loop
    for i in range(epochs):
        start_epoch = time()
        P = np.random.permutation(tot)
        if i>=lr_decay_start:
            opt.param_groups[0]['lr'] *= lr_decay_rate
        
        #train_loss = torch.tensor(0., device=device)
        train_loss = torch.tensor(0., device='cpu')
        
        # Train an epoch
        for j in range(steps):
            begin = j*batch_size
            end = (j+1)*batch_size
            
            samples, info = L.input_from_str(trlab[P[begin:end]])
            targ, _, mask = L.target(fpostr[P[begin:end]], fp=ftr, return_mz=False)
            Loss, _ = train_step(samples, targ, mask, info)
            model.global_step += 1
            #train_loss += Loss
            train_loss += Loss.detach().to('cpu')
            
            if j % 1000 == 0: print("Epoch:", i, " Step:", j, "/", steps)
            
            if torch.any(torch.isnan(train_loss)):
                print("Epoch:", i, " Step:", j, "NaN",)
                sys.exit()
        
        # Testing after training epoch
        #train_loss = train_loss.detach().to('cpu').numpy() / steps
        train_loss = train_loss.numpy() / steps

        # Val/Test loss
        test_loss = 0
        val_loss = 0
        with open(os.path.join(saved_model_path,  "val_stats"), 'w') as stats_outfile:
            #test_loss, _, _ = Testing(telab, fposte, test_point, 1)
            #test_loss, _, losses_test = Testing_np(telab, fposte, test_point, 1)
            test_loss, _, losses_test = Testing(telab, fposte, test_point, config['test_batch_size'])
            #statswriter = csv.writer(stats_outfile, delimiter='\t', quoting=csv.QUOTE_NONE)
            val_loss, varr, losses_val = Testing(vallab, fposval, val_point, config['test_batch_size'])
            #val_loss, varr, losses_val = Testing(vallab, fposval, val_point, 1, statswriter)
            

        # Result plots
        scoreDistPlot(-losses_val, "val", i)
        scoreDistPlot(-losses_test, "test", i)
        
        mirrorplot(MPIND, epoch=i, maxnorm=True, dataset="test")
        mirrorplot(MPIND2, epoch=i, maxnorm=True, dataset="test")
        
        # Results plots for worst predictions
        #losses_test_argsorted =  [k for (v, k) in sorted((v, k) for (k, v) in enumerate(losses_test))]
        #losses_val_argsorted =  [k for (v, k) in sorted((v, k) for (k, v) in enumerate(losses_val))]
        #for rank in range(min(5, len(losses_test))):
        #    print(i, rank, losses_test[losses_test_argsorted[rank]], losses_val[losses_val_argsorted[rank]])
        #    mirrorplot(losses_test_argsorted[rank], epoch=i, maxnorm=True, rank=rank, dataset="test")
        #    mirrorplot(losses_val_argsorted[rank], epoch=i, maxnorm=True, rank=rank, dataset="val")
        
        
        
        # Save checkpoint
        if svwts=='top':
            if -val_loss>currbest:
                currbest = -val_loss
                torch.save(model.state_dict(), os.path.join(saved_model_path, "ckpt_step%d_%.4f"%(model.global_step,-val_loss)))      
        elif (svwts=='all') | (svwts=='True'):
            torch.save(model.state_dict(), os.path.join(saved_model_path, "ckpt_step%d_%.4f"%(model.global_step,-val_loss)))
        torch.save(opt.state_dict(), os.path.join(saved_model_path, "opt.sd"))
        
        # Print out results
        string = ("Epoch %d; Train loss: %.4f; Val loss: %6.4f; Test loss: %6.4f; %.1f s"%(i, train_loss, -val_loss, -test_loss, time()-start_epoch))
        sys.stdout.write("\r"+string+"\n")
        
        wandb.log({"train": {"loss": train_loss, "Mean Spectral Angle" : -train_loss}, 
                  "val": {"loss": -val_loss, "loss_median": np.median(-losses_val), "Mean Spectral Angle" : -val_loss, "Median Spectral Angle" : np.median(-losses_val)}, 
                  "test": {"loss": -test_loss, "Mean Spectral Angle" : -test_loss},
                  "epoch": i
                  })
        
        #wandb.log({"train": {"loss": train_loss, "Mean Spectral Angle" : SpectralAngle(-train_loss)}, 
        #          "val": {"loss": -val_loss, "loss_median": np.median(-losses_val), "Mean Spectral Angle" : SpectralAngle(-val_loss), "Median Spectral Angle" : SpectralAngle(np.median(-losses_val))}, 
        #          "test": {"loss": -test_loss, "Mean Spectral Angle" : SpectralAngle(-test_loss)},
        #          "epoch": i
        #          })
        
        #sys.exit()
        
    model.to("cpu")

def scoreDistPlot(losses, dataset, epoch=0):
    fig, ax = plt.subplots()
    ax.hist(losses, 30, histtype='bar', color='blue') #density=True,
    wandb.log({"cs_dist_plot_" + dataset: wandb.Image(plt),
               "epoch": epoch})
    plt.close()

def mirrorplot(iloc=0, epoch=0, maxnorm=True, save=True, rank=-1, dataset="test"):
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
    
train(
      config['epochs'], 
      batch_size=config['batch_size'], 
      lr_decay_start=config['lr_decay_start'], 
      lr_decay_rate=config['lr_decay_rate'], 
      svwts=config['svwts']
)