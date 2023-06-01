function [JR,JRK] = convenge_condition (R_new,R_old,C,z_old,tk)

% JR & JRk =\sum_{i,j}<RiCij - RjCji, zij>
% JR = \sum_{i}<Ri, \sum_{j}(zij-zji)Cij'> 
%%
K = size(R_new,3);

JR = 0;
JRk = 0;
for  i = 1:K
    sumZ = 0;
    for j = 1:K
        sumZ  = sumZ + (z_old(:,i,j) - z_old(:,j,i))*C(:,i,j)';
    end
    JR  = JR + sum(dot(R_new(:,:,i),sumZ));
    JRk = JRk + sum(dot(R_old(:,:,i),sumZ));
end


% <Grad_R, R_new - R_old >
Grad_R = 0;
for i =1:K
    Grad_Ri = zeros(3,3);
    for j = 1:K
        if j ~= i
            Grad_Ri = Grad_Ri + (z_old(:,i,j) - z_old(:,j,i))*C(:,i,j)'; % 2019-6-26
        end
    end
    Grad_R = Grad_R + sum(dot(Grad_Ri,R_new(:,:,i)-R_old(:,:,i)));
end

% || R-R_old ||_F^2
R_Rk = 0;
for i = 1:K
    R_Rk = R_Rk + (norm(R_new(:,:,i)-R_old(:,:,i), 'fro'))^2;
end
R_Rk  =  R_Rk /(2*tk);
if R_Rk ==Inf
    R_Rk =0
end

JRK = JRk + Grad_R +R_Rk;




    
    
    