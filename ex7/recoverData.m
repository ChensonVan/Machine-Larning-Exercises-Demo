function X_rec = recoverData(Z, U, K)
%RECOVERDATA Recovers an approximation of the original data when using the 
%projected data
%   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
%   original data that has been reduced to K dimensions. It returns the
%   approximate reconstruction in X_rec.
%

% You need to return the following variables correctly.
m = size(Z, 1);
n = size(U, 1);
X_rec = zeros(m, n);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the approximation of the data by projecting back
%               onto the original space using the top K eigenvectors in U.
%
%               For the i-th example Z(i,:), the (approximate)
%               recovered data for dimension j is given as follows:
%                    v = Z(i, :)';
%                    recovered_j = v' * U(j, 1:K)';
%
%               Notice that U(j, 1:K) is a row vector.'
%               

% Solution - 1
U_reduce = U(:, 1:K);
X_rec = Z * U_reduce';


% Solution - 2
% for i = 1:m,
%     X_rec(i, :) = (U_reduce * Z(i, :)')';
%                 = Z(i, :) * U_reduce'
% end


% =============================================================

end
