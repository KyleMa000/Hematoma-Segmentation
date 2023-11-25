# Author: Kyle Ma @ BCIL 
# Created: 05/02/2023
# Implementation of Automated Hematoma Segmentation

# from torchvision sourcecode with modification
import torch
from torch import Tensor
from typing import List, Tuple
from torch.nn.functional import grid_sample
import torchvision.transforms.functional as TF

def _create_identity_grid(size: List[int]) -> Tensor:
    hw_space = [torch.linspace((-s + 1) / s, (s - 1) / s, s) for s in size]
    grid_y, grid_x = torch.meshgrid(hw_space, indexing="ij")
    return torch.stack([grid_x, grid_y], -1).unsqueeze(0)  # 1 x H x W x 2

def _cast_squeeze_in(img: Tensor, req_dtypes: List[torch.dtype]) -> Tuple[Tensor, bool, bool, torch.dtype]:
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype

def _cast_squeeze_out(img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: torch.dtype) -> Tensor:
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            # it is better to round before cast
            img = torch.round(img)
        img = img.to(out_dtype)

    return img

def _apply_grid_transform(
    img: Tensor, grid: Tensor, mode: str
) -> Tensor:

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [grid.dtype])

    if img.shape[0] > 1:
        grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])


    img = grid_sample(img, grid, mode=mode, padding_mode="zeros", align_corners=False)

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img

def elastic_transform(
    img: Tensor,
    displacement: Tensor,
    interpolation: str = "bilinear",
) -> Tensor:

    if not (isinstance(img, torch.Tensor)):
        raise TypeError(f"img should be Tensor. Got {type(img)}")

    size = list(img.shape[-2:])
    displacement = displacement.to(img.device)

    identity_grid = _create_identity_grid(size)
    grid = identity_grid.to(img.device) + displacement
    return _apply_grid_transform(img, grid, interpolation)


def get_params(alpha: List[float], sigma: List[float], size: List[int]) -> Tensor:
    dx = torch.rand([1, 1] + size) * 2 - 1
    if sigma[0] > 0.0:
        kx = int(8 * sigma[0] + 1)
        # if kernel size is even we have to make it odd
        if kx % 2 == 0:
            kx += 1
        dx = TF.gaussian_blur(dx, [kx, kx], sigma)
    dx = dx * alpha[0] / size[0]

    dy = torch.rand([1, 1] + size) * 2 - 1
    if sigma[1] > 0.0:
        ky = int(8 * sigma[1] + 1)
        # if kernel size is even we have to make it odd
        if ky % 2 == 0:
            ky += 1
        dy = TF.gaussian_blur(dy, [ky, ky], sigma)
    dy = dy * alpha[1] / size[1]
    return torch.concat([dx, dy], 1).permute([0, 2, 3, 1])

def do_elastic_transform(tensor):
    # value as indicated in github acute hematoma repo
    alpha = 5
    sigma = 4
    _, height, width = TF.get_dimensions(tensor)
    displacement = get_params([float(alpha), float(alpha)], [float(sigma), float(sigma)], [height, width])
    output = elastic_transform(tensor, displacement)
    return output