import matplotlib.pyplot as plt
import matplotlib
import torch
import math
import numpy as np

def display_array(arr, num_rows=-1, num_cols=-1, titles=None, do_colorbars=False, set_zero_black=False, cmap=None):
    arr = arr.squeeze()
    if len(arr.shape) == 4:
        arr = arr[0,...] #Account for batchsize possibility

    assert len(arr.shape) > 1 and len(arr.shape) <= 3, "Proper shape for display must be [B, H, W]"
    if len(arr.shape)==2:
        arr = arr[np.newaxis,...]
    
    if torch.is_tensor(arr): 
        if arr.requires_grad:
            arr = arr.detach()
        if arr.is_cuda:
            arr = arr.cpu()
        arr = arr.numpy()
    
    if num_rows == -1 and num_cols == -1:
        num_cols = min(len(arr),16)
        num_rows = math.ceil(len(arr) / num_cols)
    elif num_rows == -1:
        num_rows = math.ceil(len(arr) / num_cols)
    elif num_cols == -1:
        num_cols = math.ceil(len(arr) / num_rows)
    fig, axarr = plt.subplots(ncols=num_cols, nrows=num_rows, facecolor=(1, 1, 1), figsize=[3*num_cols, 3*num_rows])

    if set_zero_black:
        if cmap:
            cmap = matplotlib.cm.get_cmap(cmap).copy()
        else:
            cmap = matplotlib.cm.get_cmap("viridis").copy()
        cmap.set_bad(color='black')

    for idx, item in enumerate(arr):
        if set_zero_black:
            item = np.ma.masked_where(item < 0.05, item)

        if num_rows > 1 and num_cols > 1:
            im = axarr[int(idx/num_cols), idx%num_cols].imshow(item, cmap=cmap, interpolation="none")
            axarr[int(idx/num_cols), idx%num_cols].set_xticks([])
            axarr[int(idx/num_cols), idx%num_cols].set_yticks([])
            if do_colorbars:
                fig.colorbar(im, ax= axarr[int(idx/num_cols), idx%num_cols])
            if titles:
                axarr[int(idx/num_cols), idx%num_cols].set_title(titles[idx])
        elif num_cols > 1:
            im = axarr[idx%num_cols].imshow(item, cmap=cmap, interpolation="none") 
            axarr[idx%num_cols].set_xticks([])
            axarr[idx%num_cols].set_yticks([])
            if do_colorbars:
                fig.colorbar(im, ax=axarr[idx%num_cols])
            if titles:
                axarr[idx%num_cols].set_title(titles[idx])
        elif num_rows > 1:
            im = axarr[int(idx/num_cols)].imshow(item, cmap=cmap, interpolation="none")
            axarr[int(idx/num_cols)].set_xticks([])
            axarr[int(idx/num_cols)].set_yticks([])
            if do_colorbars:
                fig.colorbar(im, ax=axarr)
            if titles:
                axarr[int(idx/num_cols)].set_title(titles[idx])
        else:
            im = axarr.imshow(item, cmap=cmap, interpolation="none")
            axarr.set_xticks([])
            axarr.set_yticks([])
            if do_colorbars:
                fig.colorbar(im, ax=axarr)
            if titles:
                axarr.set_title(titles[idx])
    
