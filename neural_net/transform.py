import math
import random
import numpy as np
import torch
import torchvision.transforms as transforms

from skimage import transform

# Every transform takes and returns a dict {'image': ndarray, 'mask': ndarray}.
# The random transforms each own a random.Random seeded with seed + hash(self);
# __hash__ is a per-class constant, so the streams are distinct and stable
# across processes. CrossValidator reseeds them per fold.


class ToTensor(object):
    """H x W x C numpy arrays -> C x H x W float tensors (image and mask)."""

    def __init__(self, round_mask: bool, to_float=True):
        self.round_mask = round_mask
        self.to_float = to_float

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if self.round_mask:
            mask = np.rint(mask)
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        mask = torch.from_numpy(mask.transpose((2, 0, 1)))
        if self.to_float:
            image, mask = image.float(), mask.float()
        return {'image': image, 'mask': mask}


class Normalize(object):
    """Per-channel standardization of the image tensor; the mask is untouched."""

    def __init__(self, mean, std):
        self.norm = transforms.Normalize(mean, std)

    def __call__(self, sample):
        return {'image': self.norm(sample['image']), 'mask': sample['mask']}


class RandomRotate(object):
    """With the given probability, rotate by an integer angle in [-rotation, rotation] degrees."""

    def __init__(self, probability, rotation: int, seed=None):
        assert 0 <= probability <= 1
        self.probability = probability
        self.min_rotation, self.max_rotation = -rotation, rotation
        self.r = random.Random(seed + hash(self) if seed is not None else None)

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        if self.r.random() < self.probability:
            angle = self.r.randint(self.min_rotation, self.max_rotation)
            img = transform.rotate(img, angle, preserve_range=True, mode='reflect').astype('float32')
            mask = transform.rotate(mask, angle, preserve_range=True, mode='reflect').astype('float32')
        return {'image': img, 'mask': mask}

    def __hash__(self):
        return 2


class RandomHorizontalFlip(object):
    def __init__(self, probability, seed=None):
        assert 0 <= probability <= 1
        self.probability = probability
        self.r = random.Random(seed + hash(self) if seed is not None else None)

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        if self.r.random() < self.probability:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        return {'image': img, 'mask': mask}

    def __hash__(self):
        return 3


class RandomVerticalFlip(object):
    def __init__(self, probability, seed=None):
        assert 0 <= probability <= 1
        self.probability = probability
        self.r = random.Random(seed + hash(self) if seed is not None else None)

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        if self.r.random() < self.probability:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()
        return {'image': img, 'mask': mask}

    def __hash__(self):
        return 4


class RandomShear(object):
    """With the given probability, shear by an angle drawn from U(-angle, angle) degrees."""

    def __init__(self, probability, angle: int, seed=None):
        assert 0 <= probability <= 1
        self.probability = probability
        self.max_shear = math.radians(angle)
        self.min_shear = -self.max_shear
        self.r = random.Random(seed + hash(self) if seed is not None else None)

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        if self.r.random() < self.probability:
            angle = self.r.random() * (self.max_shear - self.min_shear) + self.min_shear
            tr = transform.AffineTransform(shear=angle)
            img = transform.warp(img, tr, mode='reflect', preserve_range=True).astype('float32')
            mask = transform.warp(mask, tr, mode='reflect', preserve_range=True).astype('float32')
        return {'image': img, 'mask': mask}

    def __hash__(self):
        return 5
