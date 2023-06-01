function [Dux,Duy] = ForwardD(U)

Dux = [diff(U,1,2), U(:,1) - U(:,end)];
Duy = [diff(U,1,1); U(1,:) - U(end,:)];