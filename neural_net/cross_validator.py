import os
import math
import zlib
import json
import random
import pickle
import numpy as np
import pandas as pd
import torch

from collections import defaultdict
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms

from .dataset import SatelliteDataset
from .synthetic_dataset import SyntheticFireDataset
from .histogram_matching import ClassConditionalHistogramMatcher
from .fdm import FeatureDistributionMatcher
from .transform import Normalize
from .sampler import ShuffleSampler, MixedBatchSampler
from .dsbn import convert_to_dsbn, set_dsbn_domain
from .utils import compute_prec_recall_f1_acc, initialize_weight


def _hm_worker_init(worker_id):
    """Re-seed the stochastic histogram matcher in each DataLoader worker."""
    info = torch.utils.data.get_worker_info()
    for ds in getattr(info.dataset, 'datasets', [info.dataset]):
        if getattr(ds, 'matcher', None) is not None:
            ds.matcher.reseed(info.seed % (2 ** 32))


class CrossValidator():
    """Leave-one-fold-out cross-validation on the real folds, optionally mixing
    synthetic tiles into the training batches.

    Training regimes (mutually exclusive):
      - real only: shuffled real tiles;
      - step-matched real only (``match_ratio``): real-only batches, but as many
        steps per epoch as a mixed run at that ratio;
      - mixed (``synthetic_ratio``): every batch holds a fixed share of
        synthetic tiles;
      - synthetic only (``synth_only``): one model trained on synthetic tiles,
        evaluated on every real fold.

    Mixed-regime options: histogram matching (deterministic or stochastic,
    background optionally from real negatives), feature distribution matching
    (after HM or alone), pre-fire renders added as negatives, a domain-weighted
    loss, and domain-specific batch normalization.

    There is no validation set and no model selection: the final-epoch model
    is evaluated on the held-out fold.
    """

    # Histogram-matching reference selection (see ClassConditionalHistogramMatcher).
    HM_MIN_BURNT_FRAC = 0.01
    HM_NEIGHBORS = 10
    HM_QUANTILES = 256

    def __init__(self, groups, model_tuple, criterion_tuple, train_transforms, test_transforms,
                 master_folder, csv_path, epochs, batch_size, lr, seed, result_folder,
                 product_list, mode, process_dict, mask_intervals, height, width,
                 lr_scheduler_tuple=None, test_folds=None, grad_clip=None, real_frac=1.0,
                 synthetic_folder=None, synthetic_ratio=0.0, match_ratio=None,
                 hist_match=False, hm_stochastic=False, hm_bg_negative=False, fdm=False,
                 prefire_add=False, synth_loss_weight=None, dsbn=False, synth_only=False,
                 num_workers=0, test_every=0, probe_every_steps=0):
        self.groups = groups
        self.model_tuple = model_tuple
        self.criterion_tuple = criterion_tuple
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.master_folder = master_folder
        self.csv_path = csv_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.result_folder = result_folder
        self.product_list = product_list
        self.mode = mode
        self.process_dict = process_dict
        self.mask_intervals = mask_intervals
        self.height = height
        self.width = width
        self.lr_scheduler_tuple = lr_scheduler_tuple
        self.test_folds = test_folds
        self.grad_clip = grad_clip
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Fraction of real training scenes kept (see _generate_train_set).
        assert 0 < real_frac <= 1, 'real_frac must be in (0, 1], got %s' % real_frac
        self.real_frac = real_frac

        self.synthetic_folder = synthetic_folder
        self.synthetic_ratio = synthetic_ratio
        self.match_ratio = match_ratio
        mixing = bool(synthetic_ratio and synthetic_ratio > 0)
        assert not (mixing and match_ratio), 'synthetic_ratio and match_ratio are mutually exclusive'
        if mixing or match_ratio:
            r = synthetic_ratio if mixing else match_ratio
            assert 0 < r < 1, 'ratio must be in (0, 1), got %s' % r

        self.hist_match = hist_match
        self.hm_stochastic = hm_stochastic
        self.hm_bg_negative = hm_bg_negative
        self.fdm = fdm
        self.prefire_add = prefire_add
        self.synth_loss_weight = synth_loss_weight
        self.dsbn = dsbn
        for name, on in (('hist_match', hist_match), ('fdm', fdm), ('prefire_add', prefire_add),
                         ('synth_loss_weight', synth_loss_weight is not None), ('dsbn', dsbn)):
            assert not on or mixing, '%s requires synthetic mixing (synthetic_ratio > 0)' % name
        assert not (fdm and hm_stochastic), 'fdm is only implemented with deterministic HM'
        assert synth_loss_weight is None or synth_loss_weight > 0

        self.synth_only = synth_only
        if synth_only:
            assert not mixing and not match_ratio, 'synth_only excludes synthetic_ratio / match_ratio'
            assert synthetic_folder is not None, 'synth_only requires synthetic_folder'

        self.num_workers = num_workers
        # Diagnostic only: log test-fold metrics every N epochs (and, for
        # synth_only, every N optimizer steps). Nothing is selected from them.
        self.test_every = int(test_every or 0)
        self.probe_every_steps = int(probe_every_steps or 0)

        if test_folds is not None:
            missing = set(test_folds) - set(self.groups.keys())
            assert not missing, 'Unknown test_folds: %s' % missing

        os.makedirs(self.result_folder, exist_ok=True)

    @classmethod
    def _instantiate(cls, item_tuple):
        item_class, item_args = item_tuple
        return item_class(**item_args)

    def _real_dataset(self, areas, transform):
        return SatelliteDataset(self.master_folder, self.mask_intervals, False, self.height, self.width,
                                self.product_list, self.mode, True, transform, self.process_dict,
                                self.csv_path, areas)

    def _build_model(self):
        # Seeded so that every randomly initialized part (the whole network from
        # scratch, or the decoder on top of a pretrained encoder) is identical
        # across folds and arms.
        torch.manual_seed(self.seed)
        model = self._instantiate(self.model_tuple)
        if model.pretrained:
            print('pretrained encoder: skipping Xavier re-init (encoder=%s)' % model.encoder_name)
        else:
            model.apply(initialize_weight)
        if self.dsbn:
            n_dsbn = convert_to_dsbn(model)
            assert n_dsbn > 0, 'dsbn requires a model with BatchNorm2d layers (not segformer)'
            print('DSBN: converted %d BatchNorm2d layers to domain-specific (2 branches)' % n_dsbn)
        model = model.to(self.device)

        criterion = self._instantiate(self.criterion_tuple)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = None
        if self.lr_scheduler_tuple is not None:
            scheduler = self.lr_scheduler_tuple[0](optimizer, **self.lr_scheduler_tuple[1])
        return model, criterion, optimizer, scheduler

    def start(self):
        if self.synth_only:
            return self._start_synth_only()

        df_dict = defaultdict(list)
        ordered_keys = list(self.groups.keys())
        if self.test_folds is not None:
            ordered_keys = [k for k in ordered_keys if k in self.test_folds]

        for idx, key in enumerate(ordered_keys):
            print('=' * 60)
            print('Fold %d/%d  |  test fold: %s (%d areas)' % (idx + 1, len(ordered_keys), key, len(self.groups[key])))
            result_path = os.path.join(self.result_folder, 'fold%03d_%s' % (idx, key))
            train_set = self._generate_train_set(key)
            cm = self._start_train(train_set, self.groups[key], result_path, key)
            self._append_metrics(df_dict, key, cm)

        self._write_report(df_dict)

    def _append_metrics(self, df_dict, fold, cm):
        prec, recall, f1, acc = compute_prec_recall_f1_acc(cm)
        print('test metrics [%s] -> accuracy: %.4f | burnt precision/recall/f1: %.4f / %.4f / %.4f'
              % (fold, acc, prec[1], recall[1], f1[1]))
        df_dict['fold'].append(fold)
        df_dict['test_set'].append('_'.join(self.groups[fold]))
        df_dict['accuracy'].append(acc)
        for idy in range(len(self.mask_intervals)):
            df_dict['precision_%d' % idy].append(prec[idy])
            df_dict['recall_%d' % idy].append(recall[idy])
            df_dict['f1_%d' % idy].append(f1[idy])

    def _write_report(self, df_dict):
        pd.DataFrame(df_dict).to_csv(os.path.join(self.result_folder, 'report.csv'), index=False)
        with open(os.path.join(self.result_folder, 'folds.json'), 'w') as fp:
            json.dump(dict(self.groups), fp)

    @classmethod
    def _compute_channel_stats(cls, dataset):
        # float64 accumulation: float32 loses precision over millions of pixels.
        imgs = dataset.images
        axes = tuple(range(imgs.ndim - 1))
        mean = imgs.mean(axis=axes, dtype=np.float64)
        std = imgs.std(axis=axes, dtype=np.float64)
        std = np.where(std < 1e-7, 1.0, std)
        return mean, std

    @classmethod
    def _append_normalization(cls, dataset, norm):
        base = list(dataset.transform.transforms) if dataset.transform is not None else []
        dataset.transform = transforms.Compose(base + [norm])

    def _build_train_loader(self, train_dataset, norm):
        """Training DataLoader for one fold. Returns (loader, synth_stats), where
        synth_stats is the (mean, std) used for the synthetic domain or None."""
        bs = self.batch_size
        len_real = len(train_dataset)
        mixing = bool(self.synthetic_ratio and self.synthetic_ratio > 0)

        if not mixing and not self.match_ratio:
            sampler = ShuffleSampler(train_dataset, self.seed)
            return DataLoader(train_dataset, batch_size=bs, sampler=sampler, drop_last=False), None

        ratio = self.synthetic_ratio if mixing else self.match_ratio
        real_per_batch = min(max(int(round((1.0 - ratio) * bs)), 1), bs - 1)
        synth_per_batch = bs - real_per_batch
        steps_per_epoch = int(math.ceil(len_real / real_per_batch))

        if not mixing:
            print('step-match (ratio %.3f): %d steps/epoch (baseline would be %d), batch %d real'
                  % (ratio, steps_per_epoch, int(math.ceil(len_real / bs)), bs))
            sampler = MixedBatchSampler(len_real, 0, bs, 0, steps_per_epoch, self.seed)
            return DataLoader(train_dataset, batch_sampler=sampler), None

        synthetic_dataset = SyntheticFireDataset(self.synthetic_folder, transform=self.train_transforms,
                                                 prefire_add=self.prefire_add)
        len_synth = len(synthetic_dataset)
        synth_mean, synth_std = self._compute_channel_stats(synthetic_dataset)
        print('synthetic raw mean: %s | std: %s'
              % (np.array2string(synth_mean, precision=4), np.array2string(synth_std, precision=4)))

        if self.hist_match or self.fdm:
            # The routing (which real tile each synthetic tile is aligned to) is
            # always HM's, so FDM-alone and every HM variant target identical tiles.
            matcher = ClassConditionalHistogramMatcher(
                train_dataset.images, train_dataset.masks, synthetic_dataset.masks,
                min_burnt_frac=self.HM_MIN_BURNT_FRAC, n_neighbors=self.HM_NEIGHBORS,
                n_quantiles=self.HM_QUANTILES, seed=self.seed,
                bg_from_negatives=self.hm_bg_negative)
            if self.hist_match and self.hm_stochastic:
                synthetic_dataset.matcher = matcher
                print('hist-match [stochastic per-draw]: %s' % matcher.summary())
            elif self.hist_match:
                matcher.precompute(synthetic_dataset.images, synthetic_dataset.masks)
                print('hist-match [deterministic, cached]: %s' % matcher.summary())
            if self.fdm:
                fdm = FeatureDistributionMatcher(train_dataset.images, train_dataset.masks,
                                                 min_burnt_frac=self.HM_MIN_BURNT_FRAC)
                fdm.precompute(synthetic_dataset.images, synthetic_dataset.masks,
                               matcher.deterministic_routing())
                print('fdm [%s, cached]: %s' % ('after HM' if self.hist_match else 'alone', fdm.summary()))
                if not self.hist_match:
                    print('  routing: %s' % matcher.summary())
            # Aligned synthetic tiles live in the real intensity space.
            synth_norm = norm
            synth_stats = (np.array(norm.norm.mean), np.array(norm.norm.std))
        else:
            synth_norm = Normalize(tuple(synth_mean), tuple(synth_std))
            synth_stats = (synth_mean, synth_std)
        self._append_normalization(synthetic_dataset, synth_norm)

        print('mixed training: real %d + synth %d tiles | per-batch %d real / %d synth | %d steps/epoch | %d workers'
              % (len_real, len_synth, real_per_batch, synth_per_batch, steps_per_epoch, self.num_workers))

        concat = ConcatDataset([train_dataset, synthetic_dataset])
        sampler = MixedBatchSampler(len_real, len_synth, real_per_batch, synth_per_batch, steps_per_epoch, self.seed)
        loader_kwargs = {}
        if self.num_workers > 0:
            loader_kwargs = dict(num_workers=self.num_workers, worker_init_fn=_hm_worker_init,
                                 persistent_workers=True)
        return DataLoader(concat, batch_sampler=sampler, **loader_kwargs), synth_stats

    def _start_train(self, train_set, test_set, save_path, fold_key):
        assert not set(train_set) & set(test_set)
        os.makedirs(save_path, exist_ok=True)

        n_reseed = self._reseed_transforms(fold_key)
        print('augmentation RNGs reseeded for fold %s: %d transforms' % (fold_key, n_reseed))
        print('areas -> train: %d | test: %d' % (len(train_set), len(test_set)))

        train_dataset = self._real_dataset(train_set, self.train_transforms)
        test_dataset = self._real_dataset(test_set, self.test_transforms)
        print('tiles -> train: %d | test: %d' % (len(train_dataset), len(test_dataset)))

        # Normalization statistics come from the training tiles only.
        norm_mean, norm_std = self._compute_channel_stats(train_dataset)
        print('norm mean: %s | std: %s'
              % (np.array2string(norm_mean, precision=4), np.array2string(norm_std, precision=4)))
        norm = Normalize(tuple(norm_mean), tuple(norm_std))
        self._append_normalization(train_dataset, norm)
        self._append_normalization(test_dataset, norm)

        test_sampler = ShuffleSampler(test_dataset, self.seed)
        train_loader, synth_stats = self._build_train_loader(train_dataset, norm)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, sampler=test_sampler, drop_last=False)

        model, criterion, optimizer, scheduler = self._build_model()
        print('training for %d epochs on %s' % (self.epochs, self.device))
        self._train(model, criterion, optimizer, train_loader, scheduler, test_loader=test_loader)
        cm = self._evaluate(model, test_loader)

        torch.save(model.state_dict(), os.path.join(save_path, fold_key + '_model.pt'))
        info = {'cm': cm, 'train_set': train_set, 'test_set': test_set,
                'norm_mean': norm_mean, 'norm_std': norm_std}
        if synth_stats is not None:
            info['synth_norm_mean'], info['synth_norm_std'] = synth_stats
        with open(os.path.join(save_path, fold_key + '_dict.pkl'), 'wb') as f:
            pickle.dump(info, f)
        return cm

    def _start_synth_only(self):
        """Train one model on synthetic tiles only and evaluate it on every real
        fold. Synthetic tiles are standardized with their own statistics, each
        real fold with the statistics of the other six folds."""
        bs = self.batch_size
        synthetic_dataset = SyntheticFireDataset(self.synthetic_folder, transform=self.train_transforms)
        len_synth = len(synthetic_dataset)
        synth_mean, synth_std = self._compute_channel_stats(synthetic_dataset)
        self._append_normalization(synthetic_dataset, Normalize(tuple(synth_mean), tuple(synth_std)))
        steps_per_epoch = int(math.ceil(len_synth / bs))
        sampler = MixedBatchSampler(0, len_synth, 0, bs, steps_per_epoch, self.seed)
        train_loader = DataLoader(synthetic_dataset, batch_sampler=sampler)
        print('synth-only: %d synth tiles | %d steps/epoch | synth norm mean %s std %s'
              % (len_synth, steps_per_epoch, np.array2string(synth_mean, precision=4),
                 np.array2string(synth_std, precision=4)))

        model, criterion, optimizer, scheduler = self._build_model()

        # Real folds, each normalized with leave-one-out statistics.
        fold_names = list(self.groups.keys())
        raw, moments = {}, {}
        for f in fold_names:
            ds = self._real_dataset(self.groups[f], self.test_transforms)
            raw[f] = ds
            imgs = ds.images.astype('float64')
            n = imgs.shape[0] * imgs.shape[1] * imgs.shape[2]
            moments[f] = (n, imgs.sum(axis=(0, 1, 2)), (imgs ** 2).sum(axis=(0, 1, 2)))

        eval_folds = fold_names if self.test_folds is None else [f for f in fold_names if f in self.test_folds]
        loaders = {}
        for f in eval_folds:
            N = sum(moments[g][0] for g in fold_names if g != f)
            S = sum(moments[g][1] for g in fold_names if g != f)
            SS = sum(moments[g][2] for g in fold_names if g != f)
            nmean = S / N
            nstd = np.sqrt(np.maximum(SS / N - nmean ** 2, 0.0))
            nstd = np.where(nstd < 1e-7, 1.0, nstd)
            self._append_normalization(raw[f], Normalize(tuple(nmean), tuple(nstd)))
            loaders[f] = DataLoader(raw[f], batch_size=bs, shuffle=False, drop_last=False)

        print('training (synth-only) for %d epochs on %s' % (self.epochs, self.device))
        self._train(model, criterion, optimizer, train_loader, scheduler,
                    probe_loaders=(loaders if self.test_every or self.probe_every_steps else None))
        torch.save(model.state_dict(), os.path.join(self.result_folder, 'synthonly_model.pt'))

        df_dict = defaultdict(list)
        for f in eval_folds:
            self._append_metrics(df_dict, f, self._evaluate(model, loaders[f]))
        self._write_report(df_dict)

    def _reseed_transforms(self, fold_key):
        """Make each fold's augmentation stream depend only on (seed, fold), so a
        single-fold run (TEST_FOLDS) reproduces that fold of a full CV run."""
        base = self.seed * 1000003 + zlib.crc32(str(fold_key).encode())
        n = 0
        for chain in (self.train_transforms, self.test_transforms):
            for t in (getattr(chain, 'transforms', None) or []):
                r = getattr(t, 'r', None)
                if isinstance(r, random.Random):
                    r.seed(base + hash(t))
                    n += 1
        return n

    def _generate_train_set(self, test_fold):
        """Areas of every fold except ``test_fold``. With real_frac < 1, a
        stratified share of each training fold (at least one area) is kept,
        drawn with a process-stable seed on (seed, test fold, train fold), so
        every regime trains on the same areas at a given seed."""
        result = []
        n_total = 0
        for grp, areas in self.groups.items():
            if grp == test_fold:
                continue
            n_total += len(areas)
            if self.real_frac < 1:
                k = max(1, int(round(self.real_frac * len(areas))))
                rng = random.Random(self.seed * 1000003 + zlib.crc32(('%s/%s' % (test_fold, grp)).encode()))
                areas = sorted(rng.sample(areas, k))
            result.extend(areas)
        if self.real_frac < 1:
            print('real-data scarcity: kept %d/%d train scenes (REAL_FRAC %g)'
                  % (len(result), n_total, self.real_frac))
        return result

    def _weighted_loss(self, criterion, outputs, mask, domain):
        """L = (L_real + w * L_synth) / (1 + w), with each term computed on its
        own domain's half of the batch."""
        domain = domain.to(outputs.device)
        real_idx = (domain == 0).nonzero(as_tuple=True)[0]
        synth_idx = (domain == 1).nonzero(as_tuple=True)[0]
        if len(real_idx) == 0 or len(synth_idx) == 0:
            return criterion(outputs, mask)
        w = self.synth_loss_weight
        loss_real = criterion(outputs[real_idx], mask[real_idx])
        loss_synth = criterion(outputs[synth_idx], mask[synth_idx])
        return (loss_real + w * loss_synth) / (1.0 + w)

    def _probe(self, model, loader, label):
        p, r, f1, acc = compute_prec_recall_f1_acc(self._evaluate(model, loader))
        iou = f1[1] / (2 - f1[1]) if f1[1] > 0 else 0.0
        print('    [test probe @ %s] burnt P/R/F1: %.4f / %.4f / %.4f | IoU: %.4f | acc: %.4f'
              % (label, p[1], r[1], f1[1], iou, acc))
        return iou

    def _probe_folds(self, model, probe_loaders, label):
        ious = [self._probe(model, loader, '%s fold %-8s' % (label, name))
                for name, loader in probe_loaders.items()]
        print('    [test probe @ %s] MEAN over %d folds | IoU: %.4f'
              % (label, len(ious), sum(ious) / max(len(ious), 1)))

    def _train(self, model, criterion, optimizer, train_loader, scheduler, test_loader=None, probe_loaders=None):
        n_batches = max(len(train_loader), 1)
        gstep = 0
        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            for data in train_loader:
                image = data['image'].to(self.device)
                mask = data['mask'].to(self.device, dtype=torch.long)

                optimizer.zero_grad()
                if self.dsbn:
                    set_dsbn_domain(model, data['domain'])
                outputs = model(image)
                if self.synth_loss_weight is not None:
                    loss = self._weighted_loss(criterion, outputs, mask, data['domain'])
                else:
                    loss = criterion(outputs, mask)
                loss.backward()
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()
                epoch_loss += loss.item()

                gstep += 1
                if probe_loaders is not None and self.probe_every_steps \
                        and gstep % self.probe_every_steps == 0:
                    self._probe_folds(model, probe_loaders, 'step %4d (ep %d)' % (gstep, epoch + 1))
                    model.train()

            print('[epoch %3d/%3d] train loss: %.5f' % (epoch + 1, self.epochs, epoch_loss / n_batches))

            if self.test_every and (epoch + 1) % self.test_every == 0:
                if probe_loaders is not None:
                    self._probe_folds(model, probe_loaders, 'ep %3d' % (epoch + 1))
                elif test_loader is not None:
                    self._probe(model, test_loader, 'ep %3d' % (epoch + 1))

            if scheduler is not None:
                scheduler.step()

    def _evaluate(self, model, loader):
        """Pixel confusion matrix of the model's argmax prediction over ``loader``."""
        model.eval()
        if self.dsbn:
            set_dsbn_domain(model, None)
        n = len(self.mask_intervals)
        cm = np.zeros((n, n))
        with torch.no_grad():
            for data in loader:
                outputs = model(data['image'].to(self.device))
                prediction = outputs.argmax(dim=1, keepdim=True)
                m = data['mask'].numpy().astype(np.int64).ravel()
                p = prediction.cpu().numpy().astype(np.int64).ravel()
                keep = (m >= 0) & (m < n) & (p >= 0) & (p < n)
                if not keep.all():
                    m, p = m[keep], p[keep]
                cm += np.bincount(n * m + p, minlength=n * n).reshape(n, n)
        return cm
