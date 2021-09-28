function p = predict(Theta1, Theta2, X)

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

X = [ones(m, 1), X];

a1 = X;

z2 = a1*Theta1';
% add a bias node to each training example
a2 = [ones(m, 1), sigmoid(z2)];

z3 = a2*Theta2';
% a3 is our output layer that contains the probability for each label.
a3 = sigmoid(z3);

[maxval, indices] = max(a3, [], 2);
p = indices;

end
