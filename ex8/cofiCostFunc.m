function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


% Theta : nu x n
% X     : nm x n
% Y     : nm x nu
% R     : nm x nu

pred = X * Theta';  %nm x nu    '
diff = pred - Y;

% Cost Function with regularization
J = 0.5 * sum(sum((diff.^2) .* R));
J = J + (lambda * 0.5) * sum(sum(Theta.^2));    % regularized term of theta.
J = J + (lambda * 0.5) * sum(sum(X.^2));        % regularized term of x.



% calculate X
for i = 1 : num_movies,
    % the row vector of all users that have rated movie i
    idx = find(R(i, :) == 1);   % (1 * r)

    % the list of users who have rated on movie i
    Theta_temp = Theta(idx, :); % (r * n)
    Y_temp = Y(i, idx);         % (1 * r)
    X_temp = X(i, :);           % (1 * n)

    %            ((1 * n) * (n * r)     -(1 * r)) * (r * n)  = (1 * n)
    X_grad(i, :) = (X_temp * Theta_temp' - Y_temp) * Theta_temp;   %'

    % regularization
    X_grad(i, :) = X_grad(i, :) + lambda * X_temp;
end


% calculate Theta
for i = 1 : num_users,
    % the row vector of all movies that user i has rated
    idx = find(R(:, i) == 1)';   % (1 * r)      '

    Theta_temp = Theta(i, :);   % (1 * n)
    Y_temp = Y(idx, i);         % (r * 1)
    X_temp = X(idx, :);         % (r * n)

    %                ((r * n) * (n * 1)     - (r * 1)) * (r * n) = (1 * n) 
    Theta_grad(i, :) = (X_temp * Theta_temp' - Y_temp)' * X_temp; 

    % regularization
    Theta_grad(i, :) = Theta_grad(i, :) + lambda * Theta_temp;
end


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
