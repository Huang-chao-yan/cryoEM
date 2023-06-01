function [u,MPSNRALLp, SSIMALLp, FSIMALLp] = wnnm_EM(N_Img, O_Img,nSig,ph)
% nSig=0.0061;
% Par   = ParSet(nSig);   
% u = WNNM_DeNoising( N_Img, O_Img, Par );     
for k=1:size(O_Img,3)
        Par   = ParSet(nSig);   
        u(:,:,k) = WNNM_DeNoising( N_Img(:,:,k), O_Img(:,:,k), Par );                                %WNNM denoisng function
       
%          PSNR  = csnr( O_Img(:,:,k), u(:,:,k), 0, 0 );
%          fprintf( 'Estimated Image: nSig = %2.3f, PSNR = %2.2f \n\n\n', nSig, PSNR );
end
 [MPSNRALLp, SSIMALLp, FSIMALLp] = quality(real(u)*255, double(O_Img)*255);
%     PSNR = csnr( O_Img, E_Img, 0, 0 );
     fprintf( 'Estimated Image: nSig = %2.3f, PSNR = %2.2f \n\n\n', nSig, MPSNRALLp );
%     imshow(uint8(E_Img));
end