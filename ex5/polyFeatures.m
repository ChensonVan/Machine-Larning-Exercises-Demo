function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 

% SOLUTION - 1  For-Loop
% for i = 1 : numel(X),
%     basic_x = ones(1, p) * X(:)(i);
%     X_poly(i, :) = basic_x .^ [1 : p];
% end


% SOLUTION - 2  Vertoration
% X_poly = (ones(numel(X), p\\)' .* X(:)')';
X_poly = X(:) .* ones(numel(X), p);
X_poly .^= [1 : p];

% =========================================================================

end
