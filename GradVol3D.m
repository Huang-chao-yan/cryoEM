function z = GradVol3D(u)

n  = size(u, 1); 
z  = zeros(3,n, n,n); 
z(1,1:end-1,:,:) = u(2:end,:,:) - u(1:end-1,:,:);  %kernel [0;-1;1]  
z(2,:,1:end-1,:) = u(:,2:end,:) - u(:,1:end-1,:);  %kernel [0 -1 1]
z(3,:,:,1:end-1) = u(:,:,2:end) - u(:,:,1:end-1);    

% z.dx(n,:,:) = zeros(n,n); 
% z.dy(:,n,:) = zeros(n,n);
% z.dz(:,:,n) = zeros(n,n);