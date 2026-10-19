import numpy as np


class FeatureDistributionMatcher:
    """Class-conditional Feature Distribution Matching (FDM) of synthetic tiles
    onto real ones (Abramov et al., 2020, section 3.1).

    HM matches each channel's marginal independently; FDM matches the mean
    vector and covariance matrix of the pixel distribution, so it also
    transfers the cross-channel correlation. For source pixels F_s (N_s x C)
    and target pixels F_t (N_t x C):

        whiten    U_s S_s V_s* = svd(cov(F_s - mean_s))
        colorize  U_t S_t V_t* = svd(cov(F_t - mean_t))
        F_s->t = (F_s - mean_s) U_s S_s^(-1/2) S_t^(1/2) U_t^T + mean_t

    Burned and unburned pixels are matched separately, against the same real
    reference tile HM routes the synthetic tile to (see
    ``ClassConditionalHistogramMatcher.deterministic_routing``). It runs either
    after HM or on its own, and is RNG-free.
    """

    # Fewer pixels than this in a class -> the class is left unchanged.
    MIN_PIXELS = 16
    # Relative floor on source eigenvalues before inverting them.
    RCOND = 1e-8

    def __init__(self, real_images, real_masks, min_pixels=MIN_PIXELS, min_burnt_frac=0.01):
        """
        Args:
            real_images (np.ndarray): (Nr, H, W, C) raw real training tiles.
            real_masks (np.ndarray): (Nr, H, W, 1) binary masks, burned == 1.
            min_pixels (int): minimum class size for a match on either side.
            min_burnt_frac (float): burned statistics are only kept for real
                tiles at or above this burned fraction (HM's reference gate).
        """
        self.min_pixels = int(min_pixels)
        n_real = real_images.shape[0]
        self.n_channels = real_images.shape[-1]
        real_burnt = real_masks[..., 0].astype(bool)

        # Target statistics per real tile: (mean, S^(1/2) U^T).
        self.unburnt = [None] * n_real
        self.burnt = {}
        for i in range(n_real):
            img = real_images[i]
            m = real_burnt[i]
            unburnt_px = img[~m]
            if unburnt_px.size == 0:
                unburnt_px = img.reshape(-1, self.n_channels)
            self.unburnt[i] = self._target_stats(unburnt_px)
            if m.mean() >= min_burnt_frac:
                self.burnt[i] = self._target_stats(img[m])

        self.n_matched = 0
        self.n_skipped = 0
        self.n_floored = 0
        self.n_clipped_px = 0
        self.n_total_px = 0

    def _target_stats(self, pixels):
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, self.n_channels)
        if pixels.shape[0] < self.min_pixels:
            return None
        mu = pixels.mean(axis=0)
        centered = pixels - mu
        cov = (centered.T @ centered) / (pixels.shape[0] - 1)
        u, s, _ = np.linalg.svd(cov)
        return mu, (np.diag(np.sqrt(np.maximum(s, 0.0))) @ u.T)

    def _transform(self, pixels, target):
        """Map (n, C) source pixels onto a target's statistics, or None if the
        source is too small or degenerate to whiten."""
        src = np.asarray(pixels, dtype=np.float64)
        if src.shape[0] < self.min_pixels:
            return None
        mu_s = src.mean(axis=0)
        centered = src - mu_s
        cov_s = (centered.T @ centered) / (src.shape[0] - 1)
        u_s, s_s, _ = np.linalg.svd(cov_s)
        s_max = s_s.max()
        if not np.isfinite(s_max) or s_max <= 0:
            return None
        floor = s_max * self.RCOND
        self.n_floored += int((s_s < floor).sum())
        s_s = np.maximum(s_s, floor)

        mu_t, colour = target
        whiten = u_s @ np.diag(1.0 / np.sqrt(s_s))
        return centered @ (whiten @ colour) + mu_t

    def _match_one(self, image, mask, ref, mode, bg_ref=None):
        """Match one synthetic tile against real tile ``ref`` using HM's routing
        ``mode``: 'class' matches both classes, 'unburnt' only the background."""
        out = image.astype(np.float64, copy=True)
        self.n_total_px += image.shape[0] * image.shape[1]

        burnt = mask[..., 0].astype(bool)
        unburnt = ~burnt
        u_src = ref if bg_ref is None else bg_ref
        jobs = []
        if mode == 'class' and burnt.any():
            jobs.append((burnt, self.burnt.get(ref)))
        if unburnt.any():
            jobs.append((unburnt, self.unburnt[u_src]))

        for sel, target in jobs:
            matched = self._transform(out[sel], target) if target is not None else None
            if matched is None:
                self.n_skipped += 1
                continue
            out[sel] = matched
            self.n_matched += 1

        # An affine map can push reflectance below zero; clip and count.
        negative = out < 0.0
        if negative.any():
            self.n_clipped_px += int(negative.any(axis=-1).sum())
            np.clip(out, 0.0, None, out=out)
        return out.astype(image.dtype, copy=False)

    def precompute(self, images, masks, routing):
        """Match every synthetic tile in place, using HM's deterministic routing."""
        assert len(routing) == images.shape[0], \
            'routing has %d entries for %d synthetic tiles' % (len(routing), images.shape[0])
        for i, (ref, mode, bg_ref) in enumerate(routing):
            images[i] = self._match_one(images[i], masks[i], ref, mode, bg_ref)
        return images

    def summary(self):
        clipped = (100.0 * self.n_clipped_px / self.n_total_px) if self.n_total_px else 0.0
        return ('%d class-matches applied, %d skipped (too few pixels / no reference stats) | '
                '%d eigenvalues floored | %.3f%% of pixels clipped at 0'
                % (self.n_matched, self.n_skipped, self.n_floored, clipped))
