# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
import numpy as np
import torch
import matplotlib.pylab as plt
from nn_modules import cls_fe_dft, nn_proc
from helpers import audio
from losses import loss_functions


def main_compressor():
    # Data settings
    n_data_points = 1          # How many signals to generate and evaluate
    time_series_length = 8192
    sampling_freq = 44100.
    epochs = 5000
    # Compressor settings
    threshold = -13
    ratio = 3
    attack = 2048
    # Analysis parameters
    ft_size = 1024
    hop_size = 384
    expected_time_frames = int(np.ceil(time_series_length/float(hop_size)) + np.ceil(ft_size/float(hop_size)))
    decomposition_rank = 5
    # Initialize nn modules
    # Front-ends
    dft_analysis = cls_fe_dft.Analysis(ft_size=ft_size, hop_size=hop_size)
    dft_synthesis = cls_fe_dft.Synthesis(ft_size=ft_size, hop_size=hop_size)

    # Latent processors
    aenc = nn_proc.AutoEncoder(expected_time_frames, decomposition_rank)
    phs_aenc = nn_proc.AutoEncoder(expected_time_frames, 2)

    # Initialize optimizer
    optimizer = torch.optim.Adam(list(dft_analysis.parameters()) +
                                 list(dft_synthesis.parameters()) +
                                 list(aenc.parameters()) +
                                 list(phs_aenc.parameters()),
                                 lr=1e-4
                                 )

    # Initialize a loss functional
    objective = loss_functions.mae
    for data_point in range(n_data_points):
        # Generate data
        x = audio.synth_input_sample(np.arange(time_series_length) / sampling_freq, 1)
        x = audio.synth_input_sample(np.arange(time_series_length) / sampling_freq, 1)
        y = audio.compressor(x=x, thresh=threshold, ratio=ratio, attack=attack)

        # Reshape data
        x = x.reshape(1, time_series_length)
        y = y.reshape(1, time_series_length)

        x_cuda = torch.autograd.Variable(torch.from_numpy(x).cuda(), requires_grad=True).float()
        y_cuda = torch.autograd.Variable(torch.from_numpy(y).cuda(), requires_grad=True).float()
        for epoch in range(epochs):

                # Forward analysis pass
                x_real, x_imag = dft_analysis.forward(x_cuda)

                # Magnitude-Phase computation
                mag = torch.norm(torch.cat((x_real, x_imag), 0), 2, dim=0).unsqueeze(0)
                phs = torch.atan2(x_imag, x_real+1e-6)

                # Processes Magnitude and phase individually
                mag_hat = aenc.forward(mag, skip_connections='sf')
                phs_hat = phs_aenc.forward(phs, skip_connections=False) + phs # <-- Slightly smoother convergence

                # Back to Real and Imaginary
                an_real = mag_hat * torch.cos(phs_hat)
                an_imag = mag_hat * torch.sin(phs_hat)

                # Forward synthesis pass
                x_hat = dft_synthesis.forward(an_real, an_imag)

                # Reconstruction term plus regularization -> Slightly less wiggly waveform
                loss = objective(x_hat, y_cuda) + 4e-3*mag.norm(1)
                print(loss.data[0])

                # Opt
                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm(list(dft_analysis.parameters()) +
                                              list(dft_synthesis.parameters()),
                                              max_norm=1., norm_type=1)
                optimizer.step()

                if epoch % 500 == 0:
                    # Show magnitude data
                    plt.figure(1)
                    plt.imshow(mag.data.cpu().numpy()[0, :, :].T, aspect='auto', origin='lower')
                    plt.title('Initial magnitude')

                    plt.figure(2)  # <---- Check this out! Some "sub-harmonic" content is generated for the compressor if the analysis weights make only small perturbations
                    plt.imshow(mag_hat.data.cpu().numpy()[0, :, :].T, aspect='auto', origin='lower')
                    plt.title('Processed magnitude')

                    # Plot the dictionaries
                    plt.matshow(dft_analysis.conv_analysis_real.weight.data.cpu().numpy()[:, 0, :] + 1)
                    plt.title('Conv-Analysis Real')
                    plt.matshow(dft_analysis.conv_analysis_imag.weight.data.cpu().numpy()[:, 0, :])
                    plt.title('Conv-Analysis Imag')
                    plt.matshow(dft_synthesis.conv_synthesis_real.weight.data.cpu().numpy()[:, 0, :])
                    plt.title('Conv-Synthesis Real')
                    plt.matshow(dft_synthesis.conv_synthesis_imag.weight.data.cpu().numpy()[:, 0, :])
                    plt.title('Conv-Synthesis Imag')
                    plt.show(block=True)

                    # Numpy conversion and plotting
                    plt.figure(7)
                    x_hat_np = x_hat.data.cpu().numpy()
                    x_cuda_np = x_cuda.data.cpu().numpy()
                    y_cuda_np = y_cuda.data.cpu().numpy()
                    plt.plot(x_cuda_np[0, :], label='Original Signal')
                    plt.plot(x_hat_np[0, :], label='Estimated Signal')
                    plt.plot(y_cuda_np[0, :], label='Target Signal')
                    plt.legend()
                    plt.show()

    return None


if __name__ == "__main__":
    np.random.seed(218)
    torch.manual_seed(218)
    torch.cuda.manual_seed(218)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    main_compressor()

# EOF