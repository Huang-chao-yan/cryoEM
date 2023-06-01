function w = Divz3D(z)

zx(:,:,:) = z(1,:,:,:); % x-axis
zy(:,:,:) = z(2,:,:,:); % y-axis
zz(:,:,:) = z(3,:,:,:); % z-axis
n  = size(zx, 1); 

tmp1 = zx(1,:,:); tmp2 = zy(:,1,:); tmp3 = zz(:,:,1); 

tmp1(2:n-1,:,:) = zx(2:n-1,:, :) - zx(1:n-2, :, :);  %kernel [0;-1;1]  
tmp2(:,2:n-1,:) = zy(:,2:n-1, :) - zy(:,1:n-2,  :);  %kernel [0;-1;1]  
tmp3(:,:,2:n-1) = zz(:, :,2:n-1) - zz(:,  :,1:n-2);  

tmp1(n,:,:) = -zx(n-1,:,:); 
tmp2(:,n,:) = -zy(:,n-1,:); 
tmp3(:,:,n) = -zz(:,:,n-1); 

w = tmp1 + tmp2 + tmp3; 