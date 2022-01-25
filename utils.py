import matplotlib.pyplot as plt
import torch
import math

def display_array(arr, num_rows=-1, num_cols=-1, titles=None, do_colorbars=False):
    arr = arr.squeeze()
    assert len(arr.shape) == 3, "Proper shape for display must be [B, H, W]"
    
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

    for idx, item in enumerate(arr):
        if num_rows > 1 and num_cols > 1:
            im = axarr[int(idx/num_cols), idx%num_cols].imshow(item, cmap="gray")
            axarr[int(idx/num_cols), idx%num_cols].set_xticks([])
            axarr[int(idx/num_cols), idx%num_cols].set_yticks([])
            if do_colorbars:
                fig.colorbar(im, ax= axarr[int(idx/num_cols), idx%num_cols])
            if titles:
                axarr[int(idx/num_cols), idx%num_cols].set_title(titles[idx])
        elif num_cols > 1:
            im = axarr[idx%num_cols].imshow(item, cmap="gray")
            axarr[idx%num_cols].set_xticks([])
            axarr[idx%num_cols].set_yticks([])
            if do_colorbars:
                fig.colorbar(im, ax=axarr[idx%num_cols])
            if titles:
                axarr[idx%num_cols].set_title(titles[idx])
        elif num_rows > 1:
            im = axarr[int(idx/num_cols)].imshow(item, cmap="gray")
            axarr[int(idx/num_cols)].set_xticks([])
            axarr[int(idx/num_cols)].set_yticks([])
            if do_colorbars:
                fig.colorbar(im, ax=axarr)
            if titles:
                axarr[int(idx/num_cols)].set_title(titles[idx])
        else:
            im = axarr.imshow(item, cmap="gray")
            axarr.set_xticks([])
            axarr.set_yticks([])
            if do_colorbars:
                fig.colorbar(im, ax=axarr)
            if titles:
                axarr.set_title(titles[idx])
    
