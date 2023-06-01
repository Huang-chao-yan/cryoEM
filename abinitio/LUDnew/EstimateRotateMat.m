function est_rots = EstimateRotateMat(C,ref_rot,Rinit,t)


%% Object:
%  min 1/2|| RiCij - RCji||_2^2
%  s.t  Ri'Ri = I
% Ri = Ri^k - t*\sum (RiCij -RjCji)Cij'
%% Solution:  Proximal Gradient Method (PGM)

TOL    = 1.0e-14;
K      = size(C,3);
X      = zeros(3,K,K);
X(1:2,:,:) = C;
C = X;

if ~isvar('t'), t = 0.0075; end
if ~isvar('Rinit')
    Rinit   = rand(3,3,K);
    for i = 1:K
        [U,~,V]      = svd(Rinit(:,:,i));
        Rinit(:,:,i) = U * V';
    end
end
R_old = Rinit; 
R_new = Rinit;

for iter = 1:100
    for i =1:K
        Grad_Ri = zeros(3,3);
        for j = 1:K
            tmp     = R_old(:,:,i) *C(:,i,j)-R_old(:,:,j) *C(:,j,i);
            %tmp     = (tmp/max(norm(tmp,2),1e-4));
            %tmp     = sign(tmp)/sqrt(3); 
            Grad_Ri = Grad_Ri + tmp * C(:,i,j)';
        end
        
        tmp = R_old(:,:,i) - t * Grad_Ri;
        [U,~,V] = svd(tmp,0);
        Rtmp    = U*V';
        
        Rtmp    = [Rtmp(:,1:2), cross(Rtmp(:,1), Rtmp(:,2))];
        erro = norm(Rtmp * Rtmp'-eye(3));
        if erro > TOL || abs(det(Rtmp)-1)> TOL
            [U,~,V] = svd(Rtmp);
            Rtmp(:,:,k) = U*V.';
        end
        R_new(:,:,i) = Rtmp;
    end
    
    if norm(R_new(:) - R_old(:))/norm(R_new(:))<1e-4,   break;    end
    %         t = beta*t;
    R_old = R_new;
end

est_rots = R_new; 
MSE = check_MSE(est_rots, ref_rot)



function tf = isvar(name)
%function tf = isvar(name)
% determine if "name" is a variable in the caller's workspace
if nargin < 1,    help isvar;    error arg;   end
tf = logical(1);
evalin('caller', [name ';'], 'tf=logical(0);')

