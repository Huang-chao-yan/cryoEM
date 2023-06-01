function  [JZ,JZK] = convenge_condition2(R_new,C,z_new,z_old,sk)

% JR & JRk =\sum_{i,j}<RiCij - RjCji, zij>
% JR = \sum_{i}<Ri, \sum_{j}(zij-zji)Cij'> 
%%
K = size(R_new,3);
JZ = 0;
Jzk = 0;
for  i = 1:K
    sumZ_new = 0;
    sumZ_old = 0;
    for j = 1:K
        sumZ_new  = sumZ_new + (z_new(:,i,j) - z_new(:,j,i))*C(:,i,j)';
        sumZ_old  = sumZ_old + (z_old(:,i,j) - z_old(:,j,i))*C(:,i,j)';
    end
    JZ  = JZ + sum(dot(R_new(:,:,i),sumZ_new));
    Jzk = Jzk + sum(dot(R_new(:,:,i),sumZ_old));
end


% <Grad_z, z_new - z_old >
grad_z = 0;
for i =1:K
    Grad_zij = zeros(3,1);
    for j = 1:K
        if j ~= i
            Grad_zij = Grad_zij + (R_new(:,:,i)*C(:,i,j) - R_new(:,:,j)*C(:,j,i)); % 2019-6-26
        end
    end
    grad_z = grad_z + sum(dot(Grad_zij,z_new(:,i,j)-z_old(:,i,j)));
end

% || z - zk ||_2^2
z_zk = 0;
for i = 1:K
    z_zk = z_zk + (norm(z_new(:,:,i)-z_old(:,:,i),2))^2;
end
z_zk  =  z_zk /(2*sk);
if z_zk ==Inf
    z_zk =0
end

JZK = Jzk + grad_z +z_zk;


