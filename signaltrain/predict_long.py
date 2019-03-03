"""
predicts a single long segment of audio by repeatedly predicting chunks and putting
them together

Can be run as a standalone utility routine, or as a function called from another method
"""

import numpy as np
import torch
import sys
sys.path.append('..')
import signaltrain as st

def predict_long(signal, knobs_nn, model, chunk_size, out_chunk_size, sr=44100, effect=None):

    # reshape input and knobs.  break signal up into overlapping windows
    overlap = chunk_size-out_chunk_size
    print("overlap = ",overlap)
    x = st.audio.sliding_window(signal, chunk_size, overlap=overlap)
    knobs = np.tile(knobs_nn, (x.shape[0],1))     # repeat knob settings a bunch of times

    # Move data to torch device
    x, knobs = torch.Tensor(x.astype(np.float32)), torch.Tensor(knobs.astype(np.float32))
    print("x.size() =",x.size(), ", knobs.size() =",knobs.size() )
    x_cuda, knobs_cuda = x.to(device),  knobs.to(device)

    # Do the model prediction
    y_hat, mag, mag_hat = model.forward(x_cuda, knobs_cuda)

    # Reassemble the output into one long signal
    y_pred = y_hat.cpu().detach().numpy().flatten().astype(np.float32)

    return y_pred


if __name__ == "__main__":
    import os
    import argparse
    from signaltrain.nn_modules import nn_proc

    # torch device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type('torch.FloatTensor')

    # parse command line args
    parser = argparse.ArgumentParser(description="Runs NN inference on long audio clip",\
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('checkpoint', help='Name of model checkpoint .tar file')
    parser.add_argument('audiofile', help='Name of audio file to read')
    parser.add_argument('--effect', help='Name of effect class for generating target', default='')
    args = parser.parse_args()


    # load from checkpoint
    print("Looking for checkpoint at",args.checkpoint)
    state_dict, rv = st.misc.load_checkpoint(args.checkpoint, fatal=True)
    scale_factor, shrink_factor = rv['scale_factor'], rv['shrink_factor']
    knob_names, knob_ranges = rv['knob_names'], rv['knob_ranges']
    num_knobs = len(knob_names)
    sr = rv['sr']
    chunk_size, out_chunk_size = rv['in_chunk_size'], rv['out_chunk_size']
    print(f"Effect name = {rv['effect_name']}")
    print(f"knob_names = {knob_names}")
    print(f"knob_ranges = {knob_ranges}")

    # Setup model
    model = nn_proc.st_model(scale_factor=scale_factor, shrink_factor=shrink_factor, num_knobs=num_knobs, sr=sr)
    model.load_state_dict(state_dict)   # overwrite the weights using the checkpoint


    # Input Data
    #infile="/home/shawley/datasets/signaltrain/music/Test/WindyPlaces.ITB.Mix10-2488-1644.wav"
    infile = args.audiofile
    print("reading input file ",infile)
    signal, sr = st.audio.read_audio_file(infile, sr=sr)
    print("signal.shape = ",signal.shape)

    # Choose knob settings
    #knobs_wc = np.array([0, 20.0, 95.0])  # one set of settings Ben used
    #kr = np.array([[0,1], [0,100], [0,100]] )  # knob ranges for ben's LA2A compressor
    knobs_wc = np.array([-30, 2.5, .002, .03])  # 4-knob compressor settings

    # convert to NN parameters for knobs
    kr = np.array(knob_ranges)
    knobs_nn = (knobs_wc - kr[:,0])/(kr[:,1]-kr[:,0]) - 0.5
    print("knobs_nn =",knobs_nn,", knobs_wc =",knobs_wc)
    #knobs_nn = np.random.rand(num_knobs)-0.5   # just pick some random setting
    #knobs_nn[0] = -0.45  # crank down the threshold for testing

    # Generate Target audio
    do_target = (args.effect != '')
    if do_target and (args.effect == 'comp_4c'):
        effect = st.audio.Compressor_4c()  # TODO: this should not be hard-coded
        y, _ = effect.go(signal, knobs_nn)

    # Call the predict_long routine
    print("\nCalling predict_long()...")
    y_pred = predict_long(signal, knobs_nn, model, chunk_size, out_chunk_size, sr=sr)

    print("\n...Back. Output y_pred.shape = ",y_pred.shape)

    # output files (offset pred with zeros to time-match with input & target)
    y_out = np.zeros(len(signal),dtype=np.float32)
    y_out[-len(y_pred):] = y_pred
    st.audio.write_audio_file("input.wav", signal, sr=44100)
    st.audio.write_audio_file("y_pred.wav", y_out, sr=44100)
    if do_target:
        st.audio.write_audio_file("y_target.wav", y, sr=44100)

    print("Finished.")
