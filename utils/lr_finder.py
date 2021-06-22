#! /usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'S.H. Hawley'

# executable version of lrfinder routine in signaltrain/learningrate

# imports
import numpy as np
import torch
import sys, os
sys.path.append('..')       # not something you want for generic import of this code, i.e. not up at the top of this code
import signaltrain as st
import matplotlib.pylab as plt
import argparse
from torch.utils.data import DataLoader


def lrfind(model, dataloader, optimizer, calc_loss, start=1e-6, stop=4e-3, num_lrs=150, to_screen=False):
    """ Learning Rate finder.  See leslie howard, sylvian gugger & jeremy howard's work """
    print("Running LR Find:",end="",flush=True)

    lrs, losses = [], []
    lr_tries = np.logspace(np.log10(start), np.log10(stop), num_lrs)
    ind, count, repeat = 0, 0, 3
    for x, y, knobs in dataloader:
        count+=1
        if ind >= len(lr_tries):
            break
        lr_try = lr_tries[ind]
        if count % repeat ==0:  # repeat over this many data points per lr value
            ind+=1
            print(".",sep="",end="",flush=True)
        optimizer.param_groups[0]['lr'] = lr_try

        #x_cuda, y_cuda, knobs_cuda = datagen.new()
        x_cuda, y_cuda, knobs_cuda = x.to(device), y.to(device), knobs.to(device)
        y_hat = model.forward(x_cuda, knobs_cuda)
        loss = calc_loss(y_hat.float() ,y_cuda.float()).mean()
        lrs.append(lr_try)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        if hasattr(model, 'clip_grad_norm_'):
            model.clip_grad_norm_()
        optimizer.step()

    plt.figure(1)
    #plt.semilogx(lrs,losses)
    plt.loglog(lrs,losses)
    if to_screen:
        plt.show()
    else:
        outfile = 'lrfind.png'
        plt.savefig(outfile)
        plt.close(plt.gcf())
        print("\nLR Find finished. See "+outfile)
    return




np.random.seed(218)
torch.manual_seed(218)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.manual_seed(218)
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')


# Parse command line arguments
parser = argparse.ArgumentParser(description="Trains neural network to reproduce input-output transformations.",\
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--apex', help="optimization setting to use with NVIDIA apex", default="O0")
parser.add_argument('-b', '--batch', type=int, help="batch size", default=200)
parser.add_argument('--effect', help='Name of effect to use. ("files" = search for "target_" and effect_info.ini files in path)', default="comp_4c")
parser.add_argument('--lrmax', help="max learning rate", default=1e-4) # Note: lrmax should be obtained by running lr_finder in learningrate.py
parser.add_argument('-n', '--num', type=int, help='Number of "data points" (audio clips) per epoch', default=200000)
parser.add_argument('--path', help='Directory to pull input (and maybe target) data from (default: None, means only synthesized-on-the-fly data)', default=None)
parser.add_argument('--sr', type=int, help='Sampling rate', default=44100)
parser.add_argument('-t','--target', help="type of target: chunk or stream", default="stream")
parser.add_argument('--start', type=float, help='starting learning rate for scan', default=1e-6)
parser.add_argument('--stop', type=float, help='final learning rate for scan', default=1e-1)
parser.add_argument('--screen', help='show plot on screen instead of file', action='store_true')
parser.add_argument("--sample_length", type=int, default=4096)
parser.add_argument("--num_channels", type=int, default=12)
parser.add_argument("--dilation_depth", type=int, default=10)
parser.add_argument("--num_repeat", type=int, default=1)
parser.add_argument("--kernel_size", type=int, default=2)

args = parser.parse_args()


# establish which audio effect class is being used
e = args.effect
if e == 'files':   # target outputs are given as files rather than 'live' 'plugins'
    # TODO: check to make sure there are a suitable number of 'target' files in path
    effect = st.audio.FileEffect(args.path)
elif e == 'comp_4c':
    effect = st.audio.Compressor_4c()
elif e == 'comp':
    effect = st.audio.Compressor()
elif e == 'comp_t':
    effect = st.audio.Comp_Just_Thresh()
elif e == 'comp_large':
    effect = st.audio.Compressor_4c_Large()
elif e == 'denoise':
    effect = st.audio.Denoise()
elif e == 'lowpass':
    effect = st.audio.LowPass()
elif 'VST' in e:
    print("VST plugins not integrated yet, but that would be great.")
    print("Feel free to grab Igor Gadelha' VSTRender lib to help implement this.")
    print("See https://github.com/igorgad/dpm")
    sys.exit(1)
else:
    print(f"Effect option '{e}' is not yet added")
    sys.exit(1)


effect.info()


# Initialize nn modules
model = st.nn_proc.signalNet(num_channels=args.num_channels, dilation_depth=args.dilation_depth, num_repeat=args.num_repeat, 
            num_knobs=len(effect.knob_names), kernel_size=2)
model.to(device)
chunk_size = args.sample_length
out_chunk_size = args.sample_length - (args.num_repeat * 2**args.dilation_depth -1)

optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lrmax, weight_decay=0)

datapath = args.path
synth_data = datapath is None # Are we synthesizing data or do we expect it to come from files

if synth_data:  # synthesize input & target data
    dataset = st.datasets.SynthAudioDataSet(chunk_size, effect, sr=args.sr, datapoints=args.num, y_size=out_chunk_size, augment=True)
else:           # use prerecoded files for input & target data
    dataset = st.datasets.AudioFileDataSet(chunk_size, effect, sr=args.sr,  datapoints=args.num, path=datapath+"/Train/",  y_size=out_chunk_size,
        rerun=False, augment=True, preload=True)

dataloader = DataLoader(dataset, batch_size=args.batch, num_workers=10, shuffle=True, worker_init_fn=st.datasets.worker_init)

lrfind(model, dataloader, optimizer, st.loss_functions.error_to_signal, start=args.start, stop=args.stop, to_screen=args.screen)

# EOF
