
%function []=example_test(algogradient_i,dir_w_size_i,mag_w_size_i,coef_dir_i)
%function []=example_test(algogradient1,algogradient2)
function []=example_test
d1='./cover/';

b= strcat(d1,'*.pgm');
files=dir(b);


% set payload
payload = 0.4;
% set size n
params.n=4 ; 



d2='./stego/dp_';


d2=strcat(d2,num2str(params.n),'_',num2str(payload))


% set params
params.p = -1;  % holder norm parameter
% 2n + 1 is eventually the size of second order derivative

mkdir(d2);
d2=strcat(d2,'/');

MEXstart = tic;

kl = load('polynomialkernels.mat');
kl = kl.k;    
 


%figure;

%% Run embedding simulation
for i = 1:1000 %length(files)
    %%%% Sequence
    na=num2str(i);
    c1=strcat(d1,na,'.pgm');
    
    cover=imread(c1);
    %imshow(cover)
    %s = size(params.directions);
    
    [stego, distortion,rho] = local_extrema_n(cover, payload,params);
    [stego, distortion,rho] = poly_n(cover, payload,params);

   %im=imread(strcat(d1,na,'.pgm'));
   % figure;
   %imshow(cover ~= stego); 
   %di =255*(cover ~= stego);
 
   %p = sum(sum(cover ~= stego))
   %pc = sum(sum(cover ~= stego))
   % pw= sum(sum(cover ~= stegow))
   

   s1=strcat(d2,na,'.pgm');
   imwrite(uint8(stego),s1);
   %s1=strcat(d2,na,'_diff.pgm');
   %imwrite(uint8(di),s1);

%MEXend = toc(MEXstart);
% % % fprintf(' - DONE');
% % % 
%subplot(1, 2, 1); imshow(cover); title('cover');
%subplot(1, 2, 2); imshow((double(stego) - double(cover) + 1)/2); title('embedding changes: +1 = white, -1 = black');
%fprintf('\n\nImage embedded in %.2f seconds, change rate: %.4f, distortion per pixel: %.6f\n', MEXend, sum(cover(:)~=stego(:))/numel(cover), distortion/numel(cover));

end
