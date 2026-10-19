"""7-fold geographic cross-validation of burned-area segmentation on Sentinel-2
RGB tiles, optionally mixing synthetic tiles into training.

Examples (one held-out fold, 10% of the real training scenes):

    # step-matched real-only baseline
    python cross_validation.py --model resnet_unet --imagenet --real-frac 0.1 \\
        --match-ratio 0.125 --test-folds coral

    # 1 synthetic tile in every 8, deterministic histogram matching
    python cross_validation.py --model resnet_unet --imagenet --real-frac 0.1 \\
        --synthetic-ratio 0.125 --hist-match --test-folds coral

See README.md for the full list of experiments.
"""
import argparse
import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group('data')
    g.add_argument('--real-dir', type=Path, default=REPO / 'sentinel-hub',
                   help='real dataset root, one subfolder per area (default: %(default)s)')
    g.add_argument('--synthetic-dir', type=Path, default=REPO / 'synthetic-6000',
                   help='synthetic-6000 root (default: %(default)s)')
    g.add_argument('--folds-csv', type=Path, default=REPO / 'satellite_data.csv',
                   help='area -> fold assignment (default: %(default)s)')
    g.add_argument('--log-dir', type=Path, default=REPO / 'logs',
                   help='results are written to <log-dir>/<run-name>/ (default: %(default)s)')
    g.add_argument('--run-name', help='override the generated run name')
    g.add_argument('--test-folds', help='comma-separated folds to hold out (default: all 7)')

    g = p.add_argument_group('model and optimization')
    g.add_argument('--model', choices=['resnet_unet', 'segformer'], default='resnet_unet')
    g.add_argument('--imagenet', action='store_true', help='start from an ImageNet-pretrained encoder')
    g.add_argument('--epochs', type=int, default=150)
    g.add_argument('--batch-size', type=int, default=64)
    g.add_argument('--lr', type=float, default=1e-4)
    g.add_argument('--grad-clip', type=float, default=1.0, help='max gradient norm (0 disables)')
    g.add_argument('--no-scheduler', action='store_true',
                   help='flat learning rate instead of 5-epoch warmup + cosine decay')
    g.add_argument('--seed', type=int, default=47)

    g = p.add_argument_group('training regime')
    g.add_argument('--real-frac', type=float, default=1.0,
                   help='fraction of real training areas kept, stratified per fold (default: 1)')
    x = g.add_mutually_exclusive_group()
    x.add_argument('--synthetic-ratio', type=float, default=0.0,
                   help='fraction of every training batch that is synthetic')
    x.add_argument('--match-ratio', type=float,
                   help='real-only, with as many steps/epoch as a mixed run at this ratio')
    x.add_argument('--synth-only', action='store_true',
                   help='train on synthetic tiles only, evaluate on every real fold')

    g = p.add_argument_group('synthetic-domain levers (require --synthetic-ratio)')
    g.add_argument('--hist-match', action='store_true', help='class-conditional histogram matching')
    g.add_argument('--hm-stochastic', action='store_true',
                   help='draw a histogram-matching reference per sample instead of a fixed one')
    g.add_argument('--hm-bg-negative', action='store_true',
                   help='match synthetic background to real tiles without burned area')
    g.add_argument('--fdm', action='store_true',
                   help='feature distribution matching (after HM, or alone without --hist-match)')
    g.add_argument('--prefire-add', action='store_true',
                   help='add every pre-fire render as an extra unburned synthetic tile')
    g.add_argument('--synth-loss-weight', type=float,
                   help='w in L = (L_real + w * L_synth) / (1 + w), computed per domain')
    g.add_argument('--dsbn', action='store_true',
                   help='domain-specific batch normalization (resnet_unet only)')

    g = p.add_argument_group('runtime and diagnostics')
    g.add_argument('--num-workers', type=int, default=0,
                   help='DataLoader workers (useful with --hm-stochastic)')
    g.add_argument('--test-every', type=int, default=0,
                   help='log test-fold metrics every N epochs (diagnostic only, no model selection)')
    g.add_argument('--probe-every-steps', type=int, default=0,
                   help='with --synth-only, also log every N optimizer steps')

    args = p.parse_args()
    mixing = args.synthetic_ratio > 0
    for flag in ('hist_match', 'fdm', 'prefire_add', 'dsbn'):
        if getattr(args, flag) and not mixing:
            p.error('--%s requires --synthetic-ratio > 0' % flag.replace('_', '-'))
    if args.synth_loss_weight is not None and not mixing:
        p.error('--synth-loss-weight requires --synthetic-ratio > 0')
    if (args.hm_stochastic or args.hm_bg_negative) and not args.hist_match:
        p.error('--hm-stochastic / --hm-bg-negative require --hist-match')
    if args.fdm and args.hm_stochastic:
        p.error('--fdm is only implemented with deterministic histogram matching')
    if args.dsbn and args.model == 'segformer':
        p.error('--dsbn needs BatchNorm layers; segformer uses LayerNorm')
    if args.synth_only and args.real_frac < 1:
        p.error('--real-frac has no meaning with --synth-only')
    return args


def run_name(args, test_folds):
    """<model>[_imnet]_bs<B>[_<fold>]<regime and levers>_ep<E>_seed<S>_<timestamp>"""
    tag = args.model + ('_imnet' if args.imagenet else '') + '_bs%d' % args.batch_size
    if test_folds and len(test_folds) == 1:
        tag += '_' + test_folds[0]
    if args.real_frac < 1:
        tag += '_real%g' % args.real_frac
    if args.synthetic_ratio > 0:
        tag += '_synth%g' % args.synthetic_ratio
        if args.hist_match:
            tag += '_hmstoch' if args.hm_stochastic else '_hm'
            if args.hm_bg_negative:
                tag += 'bg'
        if args.fdm:
            tag += '_fdm'
        if args.prefire_add:
            tag += '_pfadd'
        if args.synth_loss_weight is not None:
            tag += '_lw%g' % args.synth_loss_weight
        if args.dsbn:
            tag += '_dsbn'
    elif args.match_ratio:
        tag += '_match%g' % args.match_ratio
    if args.synth_only:
        tag += '_synthonly'
    if args.no_scheduler:
        tag += '_nosched'
    if args.lr != 1e-4:
        tag += '_lr%g' % args.lr
    return '%s_ep%d_seed%d_%s' % (tag, args.epochs, args.seed, datetime.now().strftime('%Y%m%d_%H%M%S'))


def main():
    args = parse_args()

    # Single GPU (DSBN routing does not survive nn.DataParallel). Must be set
    # before torch initializes CUDA.
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    import torch
    from monai.losses import GeneralizedDiceLoss
    from torch.optim import lr_scheduler
    from torchvision import transforms

    from neural_net.cross_validator import CrossValidator
    from neural_net.models import MODELS
    from neural_net.transform import (RandomHorizontalFlip, RandomRotate, RandomShear,
                                      RandomVerticalFlip, ToTensor)
    from neural_net.utils import set_seed, warmup_cosine_lambda

    set_seed(args.seed)
    print(vars(args))
    print('cuda %s | cudnn %s' % (torch.version.cuda, torch.backends.cudnn.version()))

    groups = OrderedDict()
    for fold, df in pd.read_csv(args.folds_csv).groupby('fold'):
        groups[fold] = df['folder'].tolist()
        print('fold %-8s: %d areas' % (fold, len(groups[fold])))

    test_folds = [f.strip() for f in args.test_folds.split(',')] if args.test_folds else None
    name = args.run_name or run_name(args, test_folds)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.log_dir / name
    print('RESULT PATH: %s' % result_path)

    train_transform = transforms.Compose([
        RandomRotate(0.5, 50, seed=args.seed),
        RandomVerticalFlip(0.5, seed=args.seed),
        RandomHorizontalFlip(0.5, seed=args.seed),
        RandomShear(0.5, 20, seed=args.seed),
        ToTensor(round_mask=True),
    ])
    test_transform = transforms.Compose([ToTensor(round_mask=True)])

    model_args = {'n_classes': 2, 'n_channels': 3,
                  'encoder_weights': 'imagenet' if args.imagenet else None}
    # Generalized Dice with a batch-global reduction; near-zero smoothing keeps
    # the loss exactly 0 at a perfect prediction at any batch size.
    loss_args = {'softmax': True, 'to_onehot_y': True, 'batch': True,
                 'smooth_nr': 1e-12, 'smooth_dr': 1e-12}
    scheduler = None if args.no_scheduler else (
        lr_scheduler.LambdaLR, {'lr_lambda': warmup_cosine_lambda(5, args.epochs)})

    cv = CrossValidator(
        groups, (MODELS[args.model], model_args), (GeneralizedDiceLoss, loss_args),
        train_transform, test_transform, str(args.real_dir), str(args.folds_csv),
        args.epochs, args.batch_size, args.lr, args.seed, str(result_path),
        product_list=['sentinel2'], mode='post',
        # Sentinel-2 B04, B03, B02 (red, green, blue)
        process_dict={'sentinel2': [3, 2, 1]},
        # mask values 0-36 -> class 0 (unburned), 37-255 -> class 1 (burned)
        mask_intervals=[(0, 36), (37, 255)],
        height=256, width=256,
        lr_scheduler_tuple=scheduler, test_folds=test_folds,
        grad_clip=args.grad_clip or None, real_frac=args.real_frac,
        synthetic_folder=str(args.synthetic_dir), synthetic_ratio=args.synthetic_ratio,
        match_ratio=args.match_ratio, hist_match=args.hist_match,
        hm_stochastic=args.hm_stochastic, hm_bg_negative=args.hm_bg_negative, fdm=args.fdm,
        prefire_add=args.prefire_add, synth_loss_weight=args.synth_loss_weight, dsbn=args.dsbn,
        synth_only=args.synth_only, num_workers=args.num_workers,
        test_every=args.test_every, probe_every_steps=args.probe_every_steps)
    cv.start()


if __name__ == '__main__':
    main()
