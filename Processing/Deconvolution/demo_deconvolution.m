clc, clear
close all

%% this file demo a L2 deconvolution with regularizer (gamma) as input
%  last update: 12/1/2022.

% capture
capture = loadtiff('neuron_capture.tif');
capture = single(capture);
capture = capture / max(capture(:));

% psf
psf = loadtiff('psf.tif');
psf = single(psf);
psf = psf / max(psf(:));


% define boundary
bd = 20;
crop2d = @(x)x(bd:end-bd - 1, bd:end-bd - 1,:); 


%% run deconvolution for single patch
gamma = 1e1;
deconv_img = single_layer_Tihknov_deconv(psf, capture, gamma);
deconv_img = crop2d(deconv_img);
deconv_img = deconv_img / max(deconv_img(:));

%% plot
figure,
subplot(1, 2, 1), imshow(crop2d(capture), []), subtitle('capture')
subplot(1, 2, 2), imshow(crop2d(deconv_img), []), subtitle('reconstruct')
