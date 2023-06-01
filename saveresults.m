close all;
NN = (n+1)/2;
road='./result0227/';

%ourTF = voltvadapt;
ourTF = OutPuttf;
% TV = OutPut;
TV = OutPuto;
original = ph;
erTF=0;
erTV=0;
for i=1:n
    ERTF=norm(real(ourTF(:,:,i))-original(:,:,i));
    ERTF=erTF+ERTF;
end
for i=1:n
    ERTV=norm(real(TV(:,:,i))-original(:,:,i));
    ERTV=erTV+ERTV;
end

tmp0  = real(ProjAddNoise(CTFtimesprojs(ph, ctfs, 2), SNR));
ernoise=0;
for i=1:n
    ERnoise=norm(real(tmp0(:,:,i))-original(:,:,i));
    ERnoise=ernoise+ERnoise;
end

tmp  = real(ProjAddNoise(CTFtimesprojs(squeeze(sum(ph,3)), ctfs, 2), SNR));
figure;imagesc(real(tmp)); %axis image; axis off; axis tight;
%f = figure(108); F=getframe(f); img=F.cdata;
%imwrite(img,strcat(road,fname,'K',num2str(K),'SNR',num2str(1./SNR),'obse.png')); %with white borde
set(gca,'position',[0 0 1 1]);grid off;axis normal;axis off;
saveas(gcf,[road,fname,'_K=',num2str(K),'_SNR=',num2str(1./SNR),'_Z_obse.png'],'png');


close 
tmp  = real(ProjAddNoise(CTFtimesprojs(squeeze(sum(ph,2)), ctfs, 2), SNR));
figure;imagesc(real(tmp));
set(gca,'position',[0 0 1 1]);grid off;axis normal;axis off;
saveas(gcf,[road,fname,'_K=',num2str(K),'_SNR=',num2str(1./SNR),'_Y_obse.png'],'png');

close 
tmp  = real(ProjAddNoise(CTFtimesprojs(squeeze(sum(ph,1)), ctfs, 2), SNR));
figure;imagesc(real(tmp));
set(gca,'position',[0 0 1 1]);grid off;axis normal;axis off;
saveas(gcf,[road,fname,'_K=',num2str(K),'_SNR=',num2str(1./SNR),'_X_obse.png'],'png');

close 
figure;imagesc(original(:,:,NN));
set(gca,'position',[0 0 1 1]);grid off;axis normal;axis off;
saveas(gcf,[road,fname,'_K=',num2str(K),'_SNR=',num2str(1./SNR),'_X_orig.png'],'png');

close 
figure;imagesc(real(ourTF(:,:,NN)));
set(gca,'position',[0 0 1 1]);grid off;axis normal;axis off;
saveas(gcf,[road,fname,'_K=',num2str(K),'_SNR=',num2str(1./SNR),'_X_TFfista.png'],'png');

close 
figure;imagesc(real(TV(:,:,NN)));
set(gca,'position',[0 0 1 1]);grid off;axis normal;axis off;
saveas(gcf,[road,fname,'_K=',num2str(K),'_SNR=',num2str(1./SNR),'_X_TV.png'],'png');

close 
figure;imagesc(squeeze(original(:,NN,:)));
set(gca,'position',[0 0 1 1]);grid off;axis normal;axis off;
saveas(gcf,[road,fname,'_K=',num2str(K),'_SNR=',num2str(1./SNR),'_Y_orig.png'],'png');

close 
figure;imagesc(squeeze(real(TV(:,NN,:))));
set(gca,'position',[0 0 1 1]);grid off;axis normal;axis off;
saveas(gcf,[road,fname,'_K=',num2str(K),'_SNR=',num2str(1./SNR),'_Y_TV.png'],'png');

close 
figure;imagesc(squeeze(real(ourTF(:,NN,:))));
set(gca,'position',[0 0 1 1]);grid off;axis normal;axis off;
saveas(gcf,[road,fname,'_K=',num2str(K),'_SNR=',num2str(1./SNR),'_Y_TFfista.png'],'png');

close 
figure;imagesc(squeeze(original(:,:,NN)));
set(gca,'position',[0 0 1 1]);grid off;axis normal;axis off;
saveas(gcf,[road,fname,'_K=',num2str(K),'_SNR=',num2str(1./SNR),'_Z_orig.png'],'png');

close 
figure;imagesc(squeeze(real(TV(:,:,NN))));
set(gca,'position',[0 0 1 1]);grid off;axis normal;axis off;
saveas(gcf,[road,fname,'_K=',num2str(K),'_SNR=',num2str(1./SNR),'_Z_TV.png'],'png');

close 
figure;imagesc(squeeze(real(ourTF(:,:,NN))));
set(gca,'position',[0 0 1 1]);grid off;axis normal;axis off;
saveas(gcf,[road,fname,'_K=',num2str(K),'_SNR=',num2str(1./SNR),'_Z_TFfista.png'],'png');

