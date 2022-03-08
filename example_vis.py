import os
import claims as clm
import pickle
import torch
import numpy as np
from tqdm import tqdm
from .utils import display_array
import matplotlib.pyplot as plt


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
    epochs = list(map(lambda epoch: epoch.replace("epoch:",""), os.listdir(os.path.join(root, date, model))))
    return epochs

def get_model_dset(split, date, model, use_fullsize=False, percent_target=0, datasets=None, root="/home/vib9/src/CLAIMS/results/models", get_3D_vols=False):
    config = load_obj(os.path.join(root, date, model, "config"))
    config.percent_target = percent_target

    if use_fullsize:
        config.use_halfsize = False

    try:
        _ = config.pad_slices
    except:
        config.pad_slices = 0

    try:
        _ = config.train_labels
    except:
        config.train_labels = None  

    try:
        _ = config.val_labels
    except:
        config.val_labels = None

    try:
        _ = config.fix_axis
    except:
        config.fix_axis = -1

    try:
        _ = config.background_prob
    except:
        config.background_prob = 0.01
    

    try:
        _ = config.use_2D_setup
    except:
        config.use_2D_setup = True

    if datasets:
        config.train_dsets = datasets
        config.train_dsets_exclude = None
        config.val_dsets = datasets

    if split == "train":
        dset, _ = clm.datasets.generate_datasets(config, get_3D_vols=get_3D_vols)
    else:
        _, dset = clm.datasets.generate_datasets(config, get_3D_vols=get_3D_vols)
        
    return dset

def get_saved_model(date, model, epoch, root="/home/vib9/src/CLAIMS/results/models", to_device=True):
    model_weights = os.path.join(root, date, model, f"epoch:{epoch}")
    config = load_obj(os.path.join(root, date, model, "config"))
    try:
        _ = config.no_querysupport_crossconv
    except:
        config.no_querysupport_crossconv = True
        config.no_supportsupport_crossconv = True
        config.use_attention = False

    try:
        _ = config.eps
    except:
        config.eps = 1

    try:
        _ = config.pad_slices
    except:
        config.pad_slices = 0 

    net = clm.config.get_net(config)
    net.load_state_dict(torch.load(model_weights))
    device = torch.device(f"cuda:{get_freer_gpu()}" if torch.cuda.is_available() else 'cpu')
    if to_device:
        net.to(device)
    net.eval()
    return net, device, config

def gen_example(args, net, dset, device):
    support_set, query_set = dset.__getitem__(0)
    support_images = support_set['images'].to(device=device, dtype=torch.float32)[np.newaxis,...]
    support_masks = support_set['labels'].to(device=device, dtype=torch.float32)[np.newaxis,...]
    query_image = query_set['images'].to(device=device, dtype=torch.float32)[np.newaxis,...]
    query_mask = query_set['labels'].to(device=device, dtype=torch.float32)

    if args.model_type == "UNet":
        pred = net(query_image.squeeze(1))
    else:
        pred = net(support_images, support_masks, query_image)
    dice = clm.losses.soft_dice(pred, query_mask, logits=True, binary=True)

    middle_query_image = query_image[:,:,args.pad_slices,...]
    if args.model_type == "UNet":
        middle_support_set = None
    else:
        cat_support_set = torch.cat([support_images, support_masks])
        middle_support_set = cat_support_set[:,:,args.pad_slices,...]
    #Accepts logits for pred
    clm.utils.training.display_forward_pass(dice.item(), middle_query_image, pred, query_mask, middle_support_set)


def get_multi_context_set_pred(args, axis, labels, sizes, net, dset, device, num_subjects=1, display_chart=False, show_examples=False):
    dset.fixed_axis = axis
    dset.labels = labels
    dset.limited_datasets = dset.datasets

    dice_per_context_size = np.zeros(len(sizes))

    for s in range(num_subjects):
        _, query_set = dset.__getitem__(0)

        context_size_dice = np.zeros(len(sizes))
        query_image = query_set['images'].to(device=device, dtype=torch.float32)[np.newaxis,...]
        query_mask = query_set['labels'].to(device=device, dtype=torch.float32)
        for s_idx, size in enumerate(sizes):
            dset.max_support_set = size
            support_set, _ = dset.__getitem__(0)
            support_images = support_set['images'].to(device=device, dtype=torch.float32)[np.newaxis,...]
            support_masks = support_set['labels'].to(device=device, dtype=torch.float32)[np.newaxis,...]
            pred = net(support_images, support_masks, query_image)

            dice = clm.losses.soft_dice(pred, query_mask, logits=True, binary=True)
            context_size_dice[s_idx] = -1 * dice

            middle_query_image = query_image[:,:,args.pad_slices,...]
            cat_support_set = torch.cat([support_images, support_masks])
            middle_support_set = cat_support_set[:,:,args.pad_slices,...]
            #Accepts logits for pred
            if show_examples:
                clm.utils.training.display_forward_pass(dice.item(), middle_query_image, pred, query_mask, middle_support_set)
    
        dice_per_context_size = dice_per_context_size + context_size_dice

    dice_per_context_size /= num_subjects

    if display_chart:
        plt.plot(sizes, dice_per_context_size)
    plt.show()


def predict_3D(net, img, pred_shape, support_im=None, support_ma=None):
    # preallocate our desired memory
    seg_slices = []
    for slice_idx in range(pred_shape[-3]):
        if not (support_im is None):
            reshaped_img = img.permute(0,1,3,4,2)
            sliced_img = reshaped_img[np.newaxis,...,slice_idx] 
            pred = (torch.sigmoid(net(support_im, support_ma, sliced_img).detach())>0.5)+0 #1,1,H,W
        else:
            sliced_img = img[:,slice_idx:slice_idx+1,...] 
            pred = (torch.sigmoid(net(sliced_img).detach())>0.5)+0 #1,1,H,W
        seg_slices.append(pred)
    pred = torch.cat(seg_slices, dim=1)
    return pred


def get_multi_axis_perf(args, net, dset, device, num_samples=0, labels=None, axes=[0,1,2], use_all_subjs=False, show_output=False, get_3D_vols=False, show_examples=False):
    losses = []
    for axis in axes:
        val_dices = get_val_perf(args, net, dset, device, num_samples=num_samples, labels=labels, axis=0, use_all_subjs=use_all_subjs, show_output=False, get_3D_vols=get_3D_vols, show_examples=show_examples)
        losses.append(val_dices)
    losses = np.concatenate(np.array(losses))
    val_dice_loss = np.mean(losses)
    val_dice_std = np.std(losses)
    print(f"Avg Val Hard Dice:", np.round(val_dice_loss,3))
    print(f"Stdv of Hard Dice:", np.round(val_dice_std,3))
    return val_dice_loss, val_dice_std


def get_val_perf(args, nets, model_types, dset, device, num_samples=0, labels=None, axis=-1, use_all_subjs=False, show_output=False, get_3D_vols=False, show_examples=False):
    assert not(num_samples > 0 and use_all_subjs), "Can either do samples or go through subjects, not both."
    dset.fix_axis = axis

    if use_all_subjs:
        dset.go_through_all_idxs = True
    else:
        dset.num_iterations = num_samples
    
    if labels:
        dset.labels = labels
        dset.limited_datasets = dset.datasets
    dset.return_names = True

    loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=True, num_workers=1, drop_last=True, pin_memory=True)
    epoch_val_dice_list = []

    iteration = 0
    with tqdm(total=len(loader), desc=f'Validation Loop', unit='batch') as pbar:
        with torch.no_grad():
            for (support_set, query_set, query_example_name) in loader:
                support_images = support_set['images'].to(device=device, dtype=torch.float32)
                support_masks = support_set['labels'].to(device=device, dtype=torch.float32)
                query_image = query_set['images'].to(device=device, dtype=torch.float32)
                query_mask = query_set['labels'].to(device=device, dtype=torch.float32)

                pred_list = []
                dice_list = []
                for mod_idx, mod_type in enumerate(model_types):
                    if mod_type == "UNet" and not get_3D_vols:
                        pred = nets[mod_idx](query_image.squeeze(1))
                        dice = clm.losses.soft_dice(pred, query_mask.squeeze(1), eps=args.eps, binary=True)
                    elif mod_type == "UNet" and get_3D_vols:
                        pred = predict_3D(nets[mod_idx], query_image.squeeze(1), pred_shape=query_mask.shape)
                        dice = clm.losses.soft_dice(pred, query_mask.squeeze(1), eps=args.eps, logits=False, binary=False, do3D=True)
                    elif get_3D_vols:
                        pred = predict_3D(nets[mod_idx], query_image, pred_shape=query_mask.shape, support_im=support_images, support_ma=support_masks)
                        dice = clm.losses.soft_dice(pred, query_mask.squeeze(1), eps=args.eps, logits=False, binary=False, do3D=True)
                    else:
                        pred = nets[mod_idx](support_images, support_masks, query_image)
                        dice = clm.losses.soft_dice(pred, query_mask.squeeze(1), eps=args.eps, binary=True)
                    pred_list.append(pred)
                    dice_list.append(dice)

                if show_examples:
                    for pred_idx, pred in enumerate(pred_list):
                        if get_3D_vols:
                            if args.use_2D_setup:
                                chosen_slice = int(query_image.shape[2]/2)
                            else:
                                label_amounts = np.count_nonzero(query_mask.squeeze().cpu(), axis=(1,2))
                                label_prob = label_amounts/np.sum(label_amounts) + 0.001
                                label_prob = label_prob/np.sum(label_prob)
                                chosen_slice = np.random.choice(np.arange(len(label_amounts)), p=label_prob)

                            chosen_image = query_image[:, :, chosen_slice, ...]
                            chosen_mask = query_mask[:, :, chosen_slice, ...]
                            chosen_pred = pred[:, chosen_slice:chosen_slice+1, ...]
                            slice_dice = clm.losses.soft_dice(chosen_pred, chosen_mask, eps=args.eps, logits=False, binary=False, do3D=False)
                        else:
                            chosen_image = query_image[:,:,args.pad_slices,...]
                            chosen_mask = query_mask.squeeze(1)
                            chosen_pred = pred
                            slice_dice = dice_list[pred_idx]

                        if model_types[pred_idx] == "UNet":
                            middle_support_set = None
                        else:
                            cat_support_set = torch.cat([support_images, support_masks]).cpu()
                            middle_support_set = cat_support_set[:,:,args.pad_slices,...]
                        #Accepts logits for pred
                        clm.utils.training.display_forward_pass(slice_dice.item(), chosen_image.cpu(), chosen_pred.cpu(), chosen_mask.cpu(), middle_support_set, sub_name=query_example_name)

                epoch_val_dice_list.append(dice_list)
                pbar.update(support_images.shape[0])
                iteration += 1
            epoch_val_dice_list = torch.tensor(epoch_val_dice_list)
    pbar.close()
    print(epoch_val_dice_list.shape)

    if show_output:
        for idx in range(len(nets)):
            val_dice_loss = torch.mean(epoch_val_dice_list[:,idx]).numpy().item()
            val_dice_std = torch.std(epoch_val_dice_list[:,idx]).numpy().item()
            print(f"Avg Val Hard Dice for {len(loader)} examples Net {idx}:", np.round(val_dice_loss,3))
            print(f"Stdv of Hard Dice for {len(loader)} examples Net {idx}:", np.round(val_dice_std,3))
        return val_dice_loss, val_dice_std
    else:
        return epoch_val_dice_list.numpy()


