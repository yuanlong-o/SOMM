clc, clear

%% this file would generate Zernike polynomials, each with coefficient 0 - 1.

%  update: change nan to zeros
%  last update: 12/12/2019


%% parameter
%  size should be taken from simulation design file
size_h = 750;
max_Zernike = 100;


%% define Zernike wavefront
x = linspace(-1, 1, size_h);
[X,Y] = meshgrid(x,x);
[theta,r] = cart2pol(X,Y);
idx = r<=1;
z = zeros(size(X));
p = [3, 5 : 5 + max_Zernike - 1]; % cancel the shift and defocus one
y = zernfun2(p, r(idx),theta(idx));
z_array = zeros(size_h, size_h, max_Zernike);
for k = 1 : max_Zernike
    z(idx) = y(:,k);
    z_array(:, :, k) = z;
    
%     figure, imagesc(z), axis equal
end
z_array = single(z_array);
save(sprintf('Zernike_max_%d_eff_size_%d.mat', max_Zernike, size_h), 'z_array', '-v7.3')


