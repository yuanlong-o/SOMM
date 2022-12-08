function [x1_crop] = single_layer_Tihknov_deconv(psf, observ, lambda1)
%% two-layer Tihknov deconvolution
%  Input:
%  psf_1, psf_2: psfs
%  image: sample that you want to observe
%  lambda1, lamdba2: regularization
%  ouput: 
%  x1_crop: result after deconvolution

%  last update: 8/6/2019



[Nx, Ny] = size(psf);   %Get problem size, determined by the PSF?


 % Smooth the boundary of psf
 %{
w_psfCut = 100; %10
kg = fspecial('gaussian',w_psfCut*[1,1],w_psfCut/10); %2);
crpSmth = zeros(size(psf));
crpSmth(w_psfCut+1:end-w_psfCut,w_psfCut+1:end-w_psfCut) = 1;
crpSmth = imfilter(crpSmth,kg,'same');
psf = bsxfun(@times, psf, crpSmth);
%}
% lateral pooling size
p1 = floor(Nx/2);
p2 = floor(Ny/2);

%% forward operators
pad2d = @(x)padarray(x,[p1,p2],'both'); 
crop2d = @(x)x(p1+1:end-p1,p2+1:end-p2,:); 

H1s = fft2(ifftshift(pad2d(psf)));
H1s_conj = conj(H1s);

observ = pad2d(observ);
H1for = @(x)real((ifft2(H1s.*fft2((x))))); 
H1conj = @(x)real((ifft2(H1s_conj.*fft2((x))))); 
H1tH1 = abs(H1s_conj .* H1s);
x1_denominator = 1 ./ (H1tH1 + lambda1);

x1_numerator = H1conj(observ);

%% result
x1 = real(ifft2(fft2(x1_numerator) .* x1_denominator));

% output data fadelity and regularization
% regularization_1 = lambda1 * sum(x1.^2, 'all')
% data_fadelity = sum((H1for(x1) - observ).^2, 'all')

x1_crop = crop2d(x1);

end