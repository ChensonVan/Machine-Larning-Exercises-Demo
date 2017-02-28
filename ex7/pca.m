function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).
%

% x(i) is columns vectors    
% X is matrix which is composed of row vectors  
% X = [x(1)';  x(2)';  ...;  x(m)']

% Sigma = (x(1) * x(1)' + x(2) * x(2)' + ... + x(m) * x(m)')/m;
Sigma = (1/m) * (X' * X); %' n * n
[U, S] = svd(Sigma)

% 若Sigma为m×n阵，则U为m×m阵，V为n×n阵。奇异值在S的对角线上，非负且按降序排列
% [U, S, V] = svd(Sigma);
% Sigma = U * S * V';


% =========================================================================

end
