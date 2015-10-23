

function z = PCA(X, k)
  [m, n] = size(X);
  Sigma = X * X' * (1 / m);
  [U,S,V] = svd(Sigma);
  z = U(:,1:k)' * x;
end

% PCA (not tested yet)
% reduce to k dimensions
k = 2;
% get current dimensionens of X
[m, n] = size(X);
% covariance matrix of X, mean normalized
Sigma = X*X' * (1 / m);
% singular valude decomposition on Sigma
[U,S,V] = svd(Sigma);
% use orthogonal U from column 1 to k
Ureduce = U(:,1:k);
% approximate x to the reduced dimensions
z = Ureduce' * x;


% Overall Cost Objective (not PCA), rather a test
% whether the dimensionality reduction diminished
% the information value in our data X

% Average squared projection error, e.g. how much
% information did we 'loose' from X to z
% divided by the variation of the X, which translates
% to: How much variance was retained
P = diag(S);
k_min_to_max = arrayfun(@(k) ...
  sum(P(1:k)) / sum(P) >= 0.99, ...
  [1:15])); 
% take the lowest dimension for k that minizes our
% overall goal
[J,I] = min(k_min_to_max)
k = I(1)
