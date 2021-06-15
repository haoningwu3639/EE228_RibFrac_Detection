import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import disk, remove_small_objects


def get_max_area(imbin):
    labs = label(imbin)
    rpps = sorted(regionprops(labs), key=lambda p: p.area)
    mask = labs == rpps[-1].label

    return mask

def get_thorax_mask(image):
    mask = image > -200
    mask = [ndimage.binary_fill_holes(x) for x in mask]
    mask = [get_max_area(x) for x in mask]
    mask = [mask[i] * (image[i] < -400) for i in range(len(mask))]
    mask = np.stack([ndimage.binary_fill_holes(x) for x in mask])
    return mask

def rescale(arr, target_shape, interpolation=0):
    target_shape = target_shape[::-1]
    arr = sitk.GetImageFromArray(arr.astype(np.uint8))
    old_spacing = arr.GetSpacing()
    old_shape = arr.GetSize()
    target_spacing = tuple([old_spacing[i] * old_shape[i] / target_shape[i] for i in range(len(target_shape))])

    resample = sitk.ResampleImageFilter()
    interpolator = sitk.sitkLinear if interpolation == 1 else sitk.sitkNearestNeighbor
    resample.SetInterpolator(interpolator)
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(target_shape)
    new_arr = resample.Execute(arr)

    return sitk.GetArrayFromImage(new_arr).astype(np.bool)

def get_lung_mask(image, shrink_ratio):
    mask = get_thorax_mask(image)
    old_shape = image.shape
    target_shape = tuple([round(dim * shrink_ratio) for dim in old_shape])
    mask = rescale(mask, target_shape)

    labs = label(mask)
    rpps = sorted(regionprops(labs), key=lambda p: p.area)
    mask = labs == rpps[-1].label

    if rpps[-2].area > rpps[-1].area / 2:
        mask = mask | (labs == rpps[-2].label)

    xpix = mask.sum((0, 1))
    labs = label(xpix < xpix.mean())
    xcrg = np.where(labs == labs[len(labs) // 2])[0]

    tube = []
    for chil in mask:
        chil = ndimage.binary_erosion(chil, disk(3))
        labs = label(chil)
        rpps = regionprops(labs)
        for p in rpps:
            x = int(p.centroid[-1])
            labs[labs == p.label] = 0 if x not in xcrg else p.label
        tube.append(ndimage.binary_dilation(labs > 0, disk(3)))
    tube = np.stack(tube)

    mask = mask * (tube == 0)
    mask = np.stack([ndimage.binary_closing(x, disk(10)) for x in mask])
    mask = get_max_area(mask)

    return mask

def get_lung_contour(image, shrink_ratio):
    old_shape = image.shape
    lung_mask = get_lung_mask(image, shrink_ratio)
    lung_contour = np.logical_xor(ndimage.maximum_filter(lung_mask, 10), lung_mask)
    lung_contour = rescale(lung_contour, old_shape)

    return lung_contour

def remove_non_rib_pred(pred, image, shrink_ratio):
    # Transpose the image and prediction from xyz to zyx
    pred = pred.transpose(2, 1, 0)
    image = image.transpose(2, 1, 0)

    lung_contour = get_lung_contour(image, shrink_ratio)
    pred = np.where(lung_contour, pred, 0)

    # Transpose them back
    pred = pred.transpose(2, 1, 0)
    image = image.transpose(2, 1, 0)

    return pred

def remove_low_probs(pred, prob_thresh):
    pred = np.where(pred > prob_thresh, pred, 0)

    return pred

def remove_spine_fp(pred, image, bone_thresh):
    image_bone = image > bone_thresh
    image_bone_2d = image_bone.sum(axis=-1)
    image_bone_2d = ndimage.median_filter(image_bone_2d, 10)
    image_spine = (image_bone_2d > image_bone_2d.max() // 3)
    kernel = disk(7)
    image_spine = ndimage.binary_opening(image_spine, kernel)
    image_spine = ndimage.binary_closing(image_spine, kernel)
    image_spine_label = label(image_spine)
    max_area = 0

    for region in regionprops(image_spine_label):
        if region.area > max_area:
            max_region = region
            max_area = max_region.area
    image_spine = np.zeros_like(image_spine)
    image_spine[
        max_region.bbox[0]:max_region.bbox[2],
        max_region.bbox[1]:max_region.bbox[3]
    ] = max_region.convex_image > 0

    return np.where(image_spine[..., np.newaxis], 0, pred)

def remove_small_things(pred, size_thresh):
    pred_bin = pred > 0
    pred_bin = remove_small_objects(pred_bin, size_thresh)
    pred = np.where(pred_bin, pred, 0)

    return pred
