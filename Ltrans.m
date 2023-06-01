function P=Ltrans(X)

[m,n,s]=size(X);

P{1}=X(1:m-1,:,:)-X(2:m,:,:);
P{2}=X(:,1:n-1,:)-X(:,2:n,:);
