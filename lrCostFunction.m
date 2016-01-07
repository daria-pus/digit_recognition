function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

L=lambda*eye(size(theta,1),size(theta,1));
L(1,1)=0;

J=1/m*(-y'*log(sigmoid(X*theta))-(1-y')*log(1-sigmoid(X*theta)))+lambda/2/m*(theta(2:end)')*theta(2:end);

grad=1/m*(sigmoid(X*theta)-y)'*X+1/m*(theta)'*L;

% =============================================================

grad = grad(:);

end
