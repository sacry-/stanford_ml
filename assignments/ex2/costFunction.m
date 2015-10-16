function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
% ?y(i) log(h?(x(i))) ? (1 ? y(i)) log(1 ? h?(x(i)))

m = length(y);
sig = sigmoid(X * theta);
cost = (y .* real(log(sig))) + ((1 - y) .* log(1 - sig));
J = (-(1 / m)) * sum(cost);
vectorized = sig  - y;
grad = (1 / m) * (vectorized' * X);

% =============================================================

end
