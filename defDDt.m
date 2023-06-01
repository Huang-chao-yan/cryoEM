function [D,Dt] = defDDt

D = @(U) ForwardD(U);
Dt = @(X,Y) Dive(X,Y);
