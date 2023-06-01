function [W, res] = W_weights(est_rots, C)

epsilon = 1e-8;
K = size(C,2);

epsilon = 1e-8;
X = zeros(3,K,K);
for i = 1:K
    for j =1:K
        X(1:2,i,j) = C(1:2,i,j);
    end
end
C = X;

W = ones(2*K);
for i = 1:K
    for j = i+1:K
        Zij = 2 - 2*(est_rots(:,:,i)*C(:,i,j))'*(est_rots(:,:,j)*C(:,j,i));
        Zji = 2 - 2*(est_rots(:,:,j)*C(:,j,i))'*(est_rots(:,:,i)*C(:,i,j));
        W(i,j) = 1/sqrt(Zij + epsilon^2);
        W(j,i) = 1/sqrt(Zij + epsilon^2);
    end
end
res = sum(W(:));


end