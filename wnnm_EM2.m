function [u,MPSNRALLp, SSIMALLp, FSIMALLp] = wnnm_EM2(N_Img, O_Img,nSig,ph)
% nSig=0.0061;
% Par   = ParSet(nSig);   
% u = WNNM_DeNoising( N_Img, O_Img, Par );     
% for k=1:size(O_Img,3)
       Par   = ParSet(nSig);   
%          Par   = QWNNM_ParSet(nSig);   
        u= QWNNM_DeNoising(ph, N_Img, Par );                                %WNNM denoisng function
% end
     [MPSNRALLp, SSIMALLp, FSIMALLp] = quality(real(u)*255, double(ph)*255);
     fprintf( 'Estimated Image: nSig = %2.3f, PSNR = %2.2f \n\n\n', nSig, MPSNRALLp );

end