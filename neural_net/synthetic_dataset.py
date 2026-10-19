import os
import numpy as np

from PIL import Image
from torch.utils.data.dataset import Dataset


class SyntheticFireDataset(Dataset):
    """The synthetic-6000 dataset (https://doi.org/10.5281/zenodo.23117919),
    exposed with the same interface as ``SatelliteDataset``.

    Each scene directory holds ``post_fire.png`` and ``pre_fire.png`` (RGBA,
    alpha dropped) and ``burn_mask.png`` ({0, 255}). Images are scaled to
    float32 in [0, 1]; masks are binary float32.

    With ``prefire_add`` every scene's pre-fire render is appended as an extra
    unburned tile after the post-fire tiles, doubling the pool.

    Attributes mirror ``SatelliteDataset``: ``images`` (N, H, W, 3),
    ``masks`` (N, H, W, 1), and ``transform`` applied in ``__getitem__``.
    An optional ``matcher`` (stochastic histogram matching) is attached by
    CrossValidator and applied per draw, before the transform.
    """

    POST, PRE, MASK = 'post_fire.png', 'pre_fire.png', 'burn_mask.png'

    def __init__(self, folder, transform=None, prefire_add=False):
        self.folder = folder
        self.transform = transform
        self.matcher = None

        scenes = sorted(d for d in os.listdir(folder)
                        if os.path.isfile(os.path.join(folder, d, self.POST)))
        if not scenes:
            raise RuntimeError('No synthetic scenes found under %s' % folder)

        images, masks = [], []
        for name in scenes:
            images.append(self._load_rgb(os.path.join(folder, name, self.POST)))
            mask = np.asarray(Image.open(os.path.join(folder, name, self.MASK)).convert('L')) > 127
            masks.append(mask.astype(np.float32)[..., None])

        self.n_prefire = 0
        if prefire_add:
            for name in scenes:
                img = self._load_rgb(os.path.join(folder, name, self.PRE))
                images.append(img)
                masks.append(np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32))
                self.n_prefire += 1
            print('pre-fire: added %d unburnt pre-fire tiles to %d post-fire -> pool %d'
                  % (self.n_prefire, len(scenes), len(images)))

        self.images = np.stack(images, axis=0)
        self.masks = np.stack(masks, axis=0)

    @staticmethod
    def _load_rgb(path):
        return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image, mask = self.images[idx], self.masks[idx]
        if self.matcher is not None:
            image = self.matcher.apply(idx, image, mask)
        result = {'image': image, 'mask': mask}
        if self.transform is not None:
            result = self.transform(result)
        result['domain'] = 1  # 0 = real, 1 = synthetic
        return result
