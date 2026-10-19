import numpy as np


class ClassConditionalHistogramMatcher:
    """Class-conditional histogram matching (HM) of synthetic tiles onto real ones.

    Follows the image-statistics matching of Abramov et al. (2020, "Keep it
    Simple"): each synthetic tile is mapped onto a real reference tile by
    matching per-channel intensity CDFs. Two extensions:

    1. Class-conditional matching. Burned and unburned pixels are matched to the
       reference's burned and unburned pixels separately, so the burn signature
       is transferred in its own intensity regime.
    2. Burnt-ratio-aware reference selection. A real tile qualifies as a burned
       reference only if its burned fraction is >= ``min_burnt_frac``.
       - Synthetic tile with burned pixels: the qualified real tiles with the
         closest burned fraction are its candidates (mode ``'class'``).
       - Synthetic tile without burned pixels (e.g. a pre-fire render), or no
         qualified real tile in the fold: a seeded set of real negative tiles,
         matched on the unburned class only (mode ``'unburnt'``).

    The reference is always a single real tile, never a pooled CDF, which would
    mix scenes into a multimodal target.

    Two selection policies:
    - deterministic: each synthetic tile uses its first candidate; ``precompute``
      bakes the matched tiles once per fold;
    - stochastic: ``apply`` samples a candidate on every draw.

    With ``bg_from_negatives`` the unburned class of a ``'class'`` tile is
    matched to a real negative tile instead of to the burned reference's
    unburned margin, so synthetic background follows ordinary non-fire
    landscape. It uses its own RNG, leaving the default references unchanged.

    References are stored as per-class, per-channel quantile tables, so a match
    is an argsort plus an ``np.interp``.
    """

    def __init__(self, real_images, real_masks, synth_masks,
                 min_burnt_frac=0.01, n_neighbors=10, n_quantiles=256, seed=0,
                 bg_from_negatives=False):
        """
        Args:
            real_images (np.ndarray): (Nr, H, W, C) raw real training tiles.
            real_masks (np.ndarray): (Nr, H, W, 1) binary masks, burned == 1.
            synth_masks (np.ndarray): (Ns, H, W, 1) synthetic binary masks, in
                the order of the SyntheticFireDataset tiles.
            min_burnt_frac (float): minimum burned fraction for a real tile to
                serve as a burned reference; tiles below it are negatives.
            n_neighbors (int): candidate references kept per synthetic tile.
            n_quantiles (int): resolution of the quantile tables.
            seed (int): RNG seed for reference selection.
            bg_from_negatives (bool): match the unburned class to a real negative.
        """
        self.n_quantiles = int(n_quantiles)
        self.q_grid = np.linspace(0.0, 1.0, self.n_quantiles)
        self.rng = np.random.default_rng(seed)

        real_burnt = real_masks[..., 0].astype(bool)
        synth_burnt = synth_masks[..., 0].astype(bool)
        n_real = real_images.shape[0]
        self.n_channels = real_images.shape[-1]

        real_burnt_frac = real_burnt.reshape(n_real, -1).mean(axis=1)
        self.synth_burnt_frac = synth_burnt.reshape(synth_burnt.shape[0], -1).mean(axis=1)

        self.qualified = np.flatnonzero(real_burnt_frac >= min_burnt_frac)
        self.negatives = np.flatnonzero(real_burnt_frac < min_burnt_frac)
        if self.negatives.size == 0:
            self.negatives = np.arange(n_real)
        qualified_set = set(int(i) for i in self.qualified)

        # Unburned table for every real tile, burned table for qualified tiles.
        self.real_unburnt_q = np.empty((n_real, self.n_channels, self.n_quantiles), dtype=np.float32)
        self.real_burnt_q = {}
        for i in range(n_real):
            img = real_images[i]
            m = real_burnt[i]
            unburnt = img[~m]
            if unburnt.size == 0:
                unburnt = img.reshape(-1, self.n_channels)
            self.real_unburnt_q[i] = self._channel_quantiles(unburnt)
            if i in qualified_set:
                self.real_burnt_q[i] = self._channel_quantiles(img[m])

        self.ref_candidates = []
        self.ref_mode = []
        qfrac = real_burnt_frac[self.qualified] if self.qualified.size else None
        for sf in self.synth_burnt_frac:
            if sf > 0 and self.qualified.size > 0:
                order = np.argsort(np.abs(qfrac - sf), kind='stable')[:n_neighbors]
                self.ref_candidates.append(self.qualified[order])
                self.ref_mode.append('class')
            else:
                k = min(n_neighbors, self.negatives.size)
                self.ref_candidates.append(self.rng.choice(self.negatives, size=k, replace=False))
                self.ref_mode.append('unburnt')

        self.bg_from_negatives = bool(bg_from_negatives)
        self.bg_candidates = None
        if self.bg_from_negatives and self.negatives.size:
            bg_rng = np.random.default_rng(seed + 104729)
            k = min(n_neighbors, self.negatives.size)
            self.bg_candidates = [
                bg_rng.choice(self.negatives, size=k, replace=False) if mode == 'class' else None
                for mode in self.ref_mode
            ]

    def _channel_quantiles(self, pixels):
        """(n_pixels, C) -> (C, n_quantiles) inverse-CDF table."""
        if pixels.size == 0:
            return np.zeros((self.n_channels, self.n_quantiles), dtype=np.float32)
        return np.quantile(pixels, self.q_grid, axis=0).T.astype(np.float32)

    def reseed(self, seed):
        """Re-seed the per-draw RNG (called once per DataLoader worker)."""
        self.rng = np.random.default_rng(seed)

    def _match_channel(self, source, tmpl_quantiles):
        """Map 1-D ``source`` onto the reference inverse CDF, preserving rank order."""
        n = source.size
        if n == 0:
            return source
        order = np.argsort(source, kind='stable')
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n)
        src_q = ranks / max(n - 1, 1)
        return np.interp(src_q, self.q_grid, tmpl_quantiles).astype(source.dtype)

    def _match_one(self, image, mask, ref, mode, bg_ref=None):
        """Match one synthetic tile against real tile ``ref``. In mode
        ``'unburnt'`` burned pixels are left as they are. ``bg_ref`` overrides
        the tile that supplies the unburned template."""
        out = image.copy()
        burnt = mask[..., 0].astype(bool)
        unburnt = ~burnt
        u_ref = ref if bg_ref is None else bg_ref
        for c in range(self.n_channels):
            ch = out[..., c]
            if mode == 'class' and burnt.any():
                ch[burnt] = self._match_channel(image[..., c][burnt], self.real_burnt_q[ref][c])
            if unburnt.any():
                ch[unburnt] = self._match_channel(image[..., c][unburnt], self.real_unburnt_q[u_ref][c])
        return out

    def _bg_ref(self, synth_idx, pick):
        if self.bg_candidates is None:
            return None
        cand = self.bg_candidates[synth_idx]
        return None if cand is None else int(cand[pick(cand.size)])

    def apply(self, synth_idx, image, mask):
        """Stochastic per-draw match: sample one of the tile's candidates."""
        cand = self.ref_candidates[synth_idx]
        ref = int(cand[self.rng.integers(cand.size)])
        bg_ref = self._bg_ref(synth_idx, lambda n: self.rng.integers(n))
        return self._match_one(image, mask, ref, self.ref_mode[synth_idx], bg_ref)

    def deterministic_routing(self):
        """Per synthetic tile, the ``(ref, mode, bg_ref)`` of the deterministic
        policy. FDM consumes this so it targets the same real tile as HM."""
        return [(int(self.ref_candidates[i][0]), self.ref_mode[i], self._bg_ref(i, lambda n: 0))
                for i in range(len(self.ref_mode))]

    def precompute(self, images, masks):
        """Deterministically match every synthetic tile in place."""
        for i, (ref, mode, bg_ref) in enumerate(self.deterministic_routing()):
            images[i] = self._match_one(images[i], masks[i], ref, mode, bg_ref)
        return images

    def summary(self):
        n_class = sum(1 for m in self.ref_mode if m == 'class')
        bg = ''
        if self.bg_from_negatives:
            n_bg = 0 if self.bg_candidates is None else len({
                int(c[0]) for c in self.bg_candidates if c is not None})
            bg = ' | background from %d real negatives' % n_bg
        return ('%d qualified burnt refs, %d negative refs | synth routing: %d class-conditional, '
                '%d unburnt-only%s' % (self.qualified.size, self.negatives.size,
                                       n_class, len(self.ref_mode) - n_class, bg))
