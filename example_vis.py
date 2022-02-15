import os
import claims as clm
import pickle
import torch
import numpy as np


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

def get_dates(root="/home/vib9/src/CLAIMS/results/models"):
    for date in os.listdir(root):
        print(date)

def get_models(date, root="/home/vib9/src/CLAIMS/results/models"):
    for model in os.listdir(os.path.join(root, date)):
        print(model)

def get_model_info(date, model, root="/home/vib9/src/CLAIMS/results/models"):
    print(list(map(lambda epoch: epoch.replace("epoch:",""), os.listdir(os.path.join(root, date, model)))))

def get_model_dset(date, model, datasets=None, root="/home/vib9/src/CLAIMS/results/models"):
    config = load_obj(os.path.join(root, date, model, "config"))
    
    try:
        _ = config.pad_slices
    except:
        config.pad_slices = 0    

    if datasets:
        config.train_dsets = datasets
        config.train_dsets_exclude = None
    dset, _ = clm.datasets.generate_datasets(config)
    return dset

def get_saved_model(date, model, epoch, root="/home/vib9/src/CLAIMS/results/models"):
    model_weights = os.path.join(root, date, model, f"epoch:{epoch}")
    config = load_obj(os.path.join(root, date, model, "config"))
    try:
        _ = config.no_querysupport_crossconv
    except:
        config.no_querysupport_crossconv = True
        config.no_supportsupport_crossconv = True
        config.use_attention = False

    net = clm.config.get_net(config)
    net.load_state_dict(torch.load(model_weights))
    device = torch.device(f"cuda:{get_freer_gpu()}" if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()
    return net, device, config

def gen_example(net, dset, device):
    support_set, query_set = dset.__getitem__(0)
    support_images = support_set['images'].to(device=device, dtype=torch.float32)[np.newaxis,...]
    support_masks = support_set['labels'].to(device=device, dtype=torch.float32)[np.newaxis,...]
    query_image = query_set['images'].to(device=device, dtype=torch.float32)[np.newaxis,...]
    query_mask = query_set['labels'].to(device=device, dtype=torch.float32)[np.newaxis,...]

    pred = net(support_images, support_masks, query_image)
    dice = clm.losses.soft_dice(pred, query_mask)

    clm.utils.training.display_forward_pass(dice.item(), query_image, torch.sigmoid(pred), query_mask, torch.cat([support_images, support_masks]))

def get_val_perf(args, net, dset, device, num_samples=100):
    dset.num_iterations = num_samples
    loader  = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=True, num_workers=1, drop_last=True, pin_memory=True)
    _, val_dice_loss = clm.training_funcs.eval_loop(net, args.model_type, loader, clm.losses.soft_dice, num_samples, 0, 1, device, show_output=True)
    print(f"Avg Val Dice for {num_samples} iterations:", np.round(val_dice_loss,3))


