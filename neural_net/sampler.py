import random

from torch.utils.data.sampler import Sampler


class ShuffleSampler(Sampler):
    """Seeded per-epoch shuffle over a dataset."""

    def __init__(self, data_source, seed):
        super().__init__()
        self.data_source = data_source
        self.index_list = list(range(len(data_source)))
        self.r = random.Random(seed)

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        self.r.shuffle(self.index_list)
        yield from self.index_list


class MixedBatchSampler(Sampler):
    """Batch sampler with a fixed per-batch real/synthetic split and a fixed
    number of batches per epoch.

    Operates on a ``ConcatDataset([real, synthetic])``: indices ``[0, n_real)``
    are real and ``[n_real, n_real + n_synth)`` are synthetic. Every batch holds
    ``real_per_batch`` real and ``synth_per_batch`` synthetic indices. Each pool
    is shuffled and drawn without replacement, reshuffling when exhausted, so a
    pool smaller than its per-epoch demand is oversampled.

    Decoupling ``steps_per_epoch`` from the pool size is what lets the
    real-only step-matched baseline (``synth_per_batch=0``) run as many
    gradient steps as the mixed run it is compared against.
    """

    def __init__(self, n_real, n_synth, real_per_batch, synth_per_batch, steps_per_epoch, seed):
        super().__init__()
        assert real_per_batch >= 0 and synth_per_batch >= 0
        assert real_per_batch + synth_per_batch > 0
        assert steps_per_epoch > 0
        assert real_per_batch == 0 or n_real > 0
        assert synth_per_batch == 0 or n_synth > 0
        self.n_real = n_real
        self.n_synth = n_synth
        self.real_per_batch = real_per_batch
        self.synth_per_batch = synth_per_batch
        self.steps_per_epoch = steps_per_epoch
        self.r = random.Random(seed)

    def __len__(self):
        return self.steps_per_epoch

    def _make_drawer(self, lo, hi, per_batch):
        pool = list(range(lo, hi))

        def draw():
            buf = []
            while True:
                self.r.shuffle(pool)
                for x in pool:
                    buf.append(x)
                    if len(buf) == per_batch:
                        yield buf
                        buf = []
        return draw()

    def __iter__(self):
        real_drawer = (self._make_drawer(0, self.n_real, self.real_per_batch)
                       if self.real_per_batch > 0 else None)
        synth_drawer = (self._make_drawer(self.n_real, self.n_real + self.n_synth, self.synth_per_batch)
                        if self.synth_per_batch > 0 else None)
        for _ in range(self.steps_per_epoch):
            batch = []
            if real_drawer is not None:
                batch.extend(next(real_drawer))
            if synth_drawer is not None:
                batch.extend(next(synth_drawer))
            yield batch
