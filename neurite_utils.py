import numpy as np
import scipy.ndimage

def bwdist(bwvol):
    """
    positive distance transform from positive entries in logical image
    Parameters
    ----------
    bwvol : nd array
        The logical volume
    Returns
    -------
    possdtrf : nd array
        the positive distance transform
    See Also
    --------
    bw2sdtrf
    """

    # reverse volume to run scipy function
    revbwvol = np.logical_not(bwvol)

    # get distance
    return scipy.ndimage.morphology.distance_transform_edt(revbwvol)

def bw2sdtrf(bwvol):
    """
    computes the signed distance transform from the surface between the
    binary True/False elements of logical bwvol
    Note: the distance transform on either side of the surface will be +1/-1
    - i.e. there are no voxels for which the dst should be 0.
    Runtime: currently the function uses bwdist twice. If there is a quick way to
    compute the surface, bwdist could be used only once.
    Parameters
    ----------
    bwvol : nd array
        The logical volume
    Returns
    -------
    sdtrf : nd array
        the signed distance transform
    See Also
    --------
    bwdist
    """

    # get the positive transform (outside the positive island)
    posdst = bwdist(bwvol)

    # get the negative transform (distance inside the island)
    notbwvol = np.logical_not(bwvol)
    negdst = bwdist(notbwvol)

    # combine the positive and negative map
    return posdst * notbwvol - negdst * bwvol

def bw2contour(bwvol, type='both', thr=1.01):
    """
    computes the contour of island(s) on a nd logical volume
    Parameters
    ----------
    bwvol : nd array
        The logical volume
    type : optional string
        since the contour is drawn on voxels, it can be drawn on the inside
        of the island ('inner'), outside of the island ('outer'), or both
        ('both' - default)
    Returns
    -------
    contour : nd array
        the contour map of the same size of the input
    See Also
    --------
    bwdist, bw2dstrf
    """

    # obtain a signed distance transform for the bw volume
    sdtrf = bw2sdtrf(bwvol)

    if type == 'inner':
        return np.logical_and(sdtrf <= 0, sdtrf > -thr)
    elif type == 'outer':
        return np.logical_and(sdtrf >= 0, sdtrf < thr)
    else:
        assert type == 'both', 'type should only be inner, outer or both'
        return np.abs(sdtrf) < thr
    
def seg2contour(seg, exclude_zero=True, contour_type='inner', thickness=1):
    '''
    transform nd segmentation (label maps) to contour maps
    Parameters
    ----------
    seg : nd array
        volume of labels/segmentations
    exclude_zero : optional logical
        whether to exclude the zero label.
        default True
    contour_type : string
        where to draw contour voxels relative to label 'inner','outer', or 'both'
    Output
    ------
    con : nd array
        nd array (volume) of contour maps
    See Also
    --------
    seg_overlap
    '''

    # extract unique labels
    labels = np.unique(seg)
    if exclude_zero:
        labels = np.delete(labels, np.where(labels == 0))

    # get the contour of each label
    contour_map = seg * 0
    for lab in labels:

        # extract binary label map for this label
        label_map = seg == lab

        # extract contour map for this label
        thickness = thickness + 0.01
        label_contour_map = bw2contour(label_map, type=contour_type, thr=thickness)

        # assign contour to this label
        contour_map[label_contour_map] = lab

    return contour_map


def seg_overlap(vol, seg, do_contour=True, do_rgb=True, cmap=None, thickness=1.0):
    '''
    overlap a nd volume and nd segmentation (label map)
    do_contour should be None, boolean, or contour_type from seg2contour
    not well tested yet.
    '''

    # compute contours for each label if necessary
    if do_contour is not None and do_contour is not False:
        if not isinstance(do_contour, str):
            do_contour = 'inner'
        seg = seg2contour(seg, contour_type=do_contour, thickness=thickness)

    # compute a rgb-contour map
    if do_rgb:
        if cmap is None:
            nb_labels = np.max(seg).astype(int) + 1
            colors = np.random.random((nb_labels, 3)) * 0.5 + 0.5
            colors[0, :] = [0, 0, 0]
        else:
            colors = cmap[:, 0:3]

        olap = colors[seg.flat, :]
        sf = seg.flat == 0
        for d in range(3):
            olap[sf, d] = vol.flat[sf]
        olap = np.reshape(olap, vol.shape + (3, ))

    else:
        olap = seg
        olap[seg == 0] = vol[seg == 0]

    return olap


def seg_overlay(vol, seg, do_rgb=True, seg_wt=0.5, cmap=None):
    '''
    overlap a nd volume and nd segmentation (label map)
    not well tested yet.
    '''
    if do_rgb:
        if cmap is None:
            nb_labels = np.max(seg) + 1
            colors = np.random.random((nb_labels, 3)) * 0.5 + 0.5
            colors[0, :] = [0, 0, 0]
        else:
            colors = cmap[:, 0:3]
        seg_flat = colors[seg.flat, :]
        seg_rgb = np.reshape(seg_flat, vol.shape + (3, ))
        olap = seg_rgb * seg_wt + np.expand_dims(vol, -1) * (1 - seg_wt)
    else:
        olap = seg * seg_wt + vol * (1 - seg_wt)

    return olap