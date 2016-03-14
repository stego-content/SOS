


function [stego, distortion,rho] = local_extrema(cover_, payload,params)
    cover = double(cover_);
    sizeCover=size(cover);
    p=params.p;
    padSize=1;
    wetCost = 10^10;
    coverPadded = padarray(cover, [padSize padSize], 'symmetric');
    
    % calcul des dérivées secondes à partir d'une fenêtre 3*3
    k=[0,0,0 ; 1,-2,1 ; 0, 0,0];    
    d2p_dx2= conv2(coverPadded,k, 'same');    
    d2p_dy2= conv2(coverPadded,k', 'same');
    k=[0.25,-0.25,0 ; -0.25, 0.25,0; 0, 0,0];    
    d2p_dy_dx_1= conv2(coverPadded,k, 'same');
    k=[0,0,0 ; 0.25,-0.25,0 ;-0.25,0.25,0];    
    d2p_dy_dx_2= conv2(coverPadded,k, 'same');
    k=[0,0.25,-0.25; 0, -0.25,0.25;0,0,0];    
    d2p_dy_dx_3= conv2(coverPadded,k, 'same');
    k=[0,0,0; 0,0.25,-0.25; 0, -0.25, 0.25];    
    d2p_dy_dx_4= conv2(coverPadded,k, 'same');
    d2p_dy_dx = d2p_dy_dx_1+d2p_dy_dx_2 + d2p_dy_dx_3+d2p_dy_dx_4;
    
    
    rho= (abs(d2p_dx2).^(p) + abs(d2p_dy2).^(p) + (abs(d2p_dy_dx)).^(p)).^(-1/p);
    Ap = rho;
    rho = Ap(((size(Ap, 1)-sizeCover(1))/2)+1:end-((size(Ap, 1)-sizeCover(1))/2), ((size(Ap, 2)-sizeCover(2))/2)+1:end-((size(Ap, 2)-sizeCover(2))/2));
    
    

% adjust embedding costs
rrho(rho > wetCost) = wetCost; % threshold on the costs
rho(isnan(rho)) = wetCost; % if all xi{} are zero threshold the cost
rhoP1 = rho;
rhoM1 = rho;
rhoP1(cover==255) = wetCost; % do not embed +1 if the pixel has max value
rhoM1(cover==0) = wetCost; % do not embed -1 if the pixel has min value

%% Embedding simulator
stego = EmbeddingSimulator(cover, rhoP1, rhoM1, payload*numel(cover), false);
distortion_local = rho(cover~=stego);
distortion = sum(distortion_local);
end
%% --------------------------------------------------------------------------------------------------------------------------
% Embedding simulator simulates the embedding made by the best possible ternary coding method (it embeds on the entropy bound).
% This can be achieved in practice using "Multi-layered  syndrome-trellis codes" (ML STC) that are asymptotically aproaching the bound.
    function [y] = EmbeddingSimulator(x, rhoP1, rhoM1, m, fixEmbeddingChanges)
        
        n = numel(x);
        lambda = calc_lambda(rhoP1, rhoM1, m, n);
        pChangeP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
        pChangeM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
        if fixEmbeddingChanges == 1
            RandStream.setGlobalStream(RandStream('mt19937ar','seed',139187));
        else
            RandStream.setGlobalStream(RandStream('mt19937ar','Seed',sum(100*clock)));
        end
        randChange = rand(size(x));
        y = x;
        y(randChange < pChangeP1) = y(randChange < pChangeP1) + 1;
        y(randChange >= pChangeP1 & randChange < pChangeP1+pChangeM1) = y(randChange >= pChangeP1 & randChange < pChangeP1+pChangeM1) - 1;
        
        function lambda = calc_lambda(rhoP1, rhoM1, message_length, n)
            
            l3 = 1e+3;
            m3 = double(message_length + 1);
            iterations = 0;
            while m3 > message_length
                l3 = l3 * 2;
                pP1 = (exp(-l3 .* rhoP1))./(1 + exp(-l3 .* rhoP1) + exp(-l3 .* rhoM1));
                pM1 = (exp(-l3 .* rhoM1))./(1 + exp(-l3 .* rhoP1) + exp(-l3 .* rhoM1));
                m3 = ternary_entropyf(pP1, pM1);
                iterations = iterations + 1;
                if (iterations > 10)
                    lambda = l3;
                    return;
                end
            end
            
            l1 = 0;
            m1 = double(n);
            lambda = 0;
            
            alpha = double(message_length)/n;
            % limit search to 30 iterations
            % and require that relative payload embedded is roughly within 1/1000 of the required relative payload
            while  (double(m1-m3)/n > alpha/1000.0 ) && (iterations<30)
                lambda = l1+(l3-l1)/2;
                pP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
                pM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
                m2 = ternary_entropyf(pP1, pM1);
                if m2 < message_length
                    l3 = lambda;
                    m3 = m2;
                else
                    l1 = lambda;
                    m1 = m2;
                end
                iterations = iterations + 1;
            end
        end
        
        function Ht = ternary_entropyf(pP1, pM1)
            p0 = 1-pP1-pM1;
            P = [p0(:); pP1(:); pM1(:)];
            H = -((P).*log2(P));
            H((P<eps) | (P > 1-eps)) = 0;
            Ht = sum(H);
        end
    end


function r = correlationc(A,B)
    [im,jm]=size(A);
    r = 0;
    for i=1:im
        for j=1:jm
            r = r + A(i,j)*B(i,j);
        end
    end
end



% function w = iuwt2(X, d)
%     X=double(X);
%     s=size(X);
%     w = cell(d+1,1);
%     coefs=[1 4 6 4 1]
%     h= coefs/sum(coefs);
%     sh=size(h)
%     padSize=sh(2);
%     index=0:padSize-1
%     k=h;
%     n=padSize
%     cj = padarray(X, [padSize padSize], 'symmetric');        
%     for i=1:d 
%         w{i}=zeros(s(1),s(2));
%         tmp=conv2(cj,k,'same'); 
%         ck= conv2(tmp,transpose(k),'same'); 
%         wp=cj-ck; 
%         w{i} = wp(padSize+1:s(1)+padSize,padSize+1:s(2)+padSize);
%         cj=ck;
%         n=n*2-1
%         index=index*2
%         k=zeros(1,n)              
%         for j=1:padSize
%             k(index(j)+1)=h(j)
%         end
%         
%     end
%     w{d+1}=cj(padSize+1:s(1)+padSize,padSize+1:s(2)+padSize); 
% end


function w = iuwt2(X, d)
% w is an cell of d+1 arrays
% the first d arrays w{1}...w{d} are the isotropic undecimated wavelet coefs
% the last array w{d1+1} contains the coarse coefs
% the algorithm is detail in article p298
% The Undecimated Wavelet Decomposition and its Reconstruction
% Jean-Luc Starck,  Jalal Fadili, and  Fionn Murtagh
% IEEE Transactions ON IMAGE PROCESSING, VOL. 16, NO. 2, FEBRUARY 2007
    
    cj=double(X);
    s=size(X);
    w = cell(d+1,1);
    h1D=[1 4 6 4 1];
    h1D= h1D/sum(h1D);
    h1Do=h1D;
    st_=size(h1D);
    sho=st_(2); %5
    for i=1:d 
        % compute the matrices h and g 
        st_=size(h1D);
        sh=st_(2);
        delta  = zeros(sh,sh);
        g = zeros (sh,sh);
        h = zeros (sh,sh);
        delta(1,1) = 1;
        for k=1:sh
            for l=1:sh
                h(k,l) = h1D(k)*h1D(l);
                g(k,l) = delta(k,l)-h(k,l);
            end
        end
        %compute the wavelet coefs w and the coarse coefs c
        ck= conv2(cj,h,'same'); 
        %w{i}= conv2(cj,g,'same');
        w{i}= ck-cj;
        cj=ck;
        % update the h1D "à trous"
        h1D=zeros(1,sh*2-1);              
        for j=0:sho-1
            h1D(j*2^i+1)=h1Do(j+1);
        end
    end
    w{d+1}=cj;
end





function l = bresenham(e10,t)
    
    l=[0 0 ];
    xn = 0;
    yn = 0;
    e=0;
    e01 = -1;
    delta = 1; 
    out=false;
    outp=false;
    if abs(e10) > 1
        out=true;
        if e10 < 0
           outp = true;
        end
        e10 = -1/e10;
    end
    if e10 <0
        e01 = +1;
        delta = -1;
    end
    
    
    while xn < t-1
       e = e + e10;
       if abs(e) >= 0.5
           yn = yn - delta;
           e = e + e01;
       end
       xn = xn+1;
       if out
           if outp 
                l = [l ; -yn xn];

           else
                l = [l ; yn -xn];
           end
       else
           l = [l ; xn yn];
       end
    end    
end


function l=toutes_lignes_naives(N)
    l = naive_line(N);
    lp = [];
    s = size(l);
    % entre PI/4 et PI/2
    for i=1:s(1)
        dx = l(i,1);
        dy = l(i,2);
        lp = [lp ; dy dx]; 
    end
    l = [1 1; l ; lp];
    % entre PI/2 et 2PI
    lp = [];
    s = size(l);
    for i=1:s(1)
        dx = l(i,1);
        dy = l(i,2);
        lp = [lp ; dx -dy]; 
    end
    l = [l ; lp ; 0 1 ; 1 0];
    
end


function l= naive_line(N)
    l=[];
    for a=1:N-1
        for b = a+1:N-1
            if gcd(a,b) == 1
                l = [l ; a b ];
            end
        end
    end
end

function pl = parallel_lines(direction,ls)
    r = sqrt(2)*ls/2;
    lss = ceil(r);
    xcenter = lss;
    ycenter = lss;
    taillemax = 4*lss;
    b = bresenham(direction,taillemax);
    pl=cell(taillemax,1);
    count=1;
    long_min_seg = ls/2;
    % ce qui suit partitionne bien le support si 
    % si -1 <= direction <= 1 
    
    if abs(direction) <= 1
        % pour chaque ligne
        for i= 1-2*lss:3*lss
            l=zeros(2*lss);
            %l=[];
            % pour chaque pixel de cette droite
            for j=1:taillemax
                X = i+b(j,2);
                Y = 1 + b(j,1) ;
                rr = (X-0.5-xcenter)^2 + (Y-0.5-ycenter)^2;
                if (X>0 && X < 2*lss+1 & Y>0 & Y < 2*lss+1 & ( rr< r^2))
                    l(X,Y)=1;
                end
            end
            
            %%%% ERREUR SUR LE TEST
            if sum(sum(l)) > 0
                pl{count}= l;
                count = count+1;
            end
        end
    else
        if direction <0
            for j= 1-2*lss:3*lss
                l=zeros(2*lss);
                %l=[];
                % pour chaque pixel de cette droite
                for i=1:taillemax
                    X = 1+b(i,2);
                    Y = j + b(i,1);
                    rr = (X-0.5-xcenter)^2 + (Y-0.5-ycenter)^2;
                    if (X>0 && X < 2*lss+1 & Y>0 & Y < 2*lss+1 & ( rr< r^2))
                        l(X,Y)=1;%l = [l ; X Y];
                    end
                end
                if sum(sum(l)) > 0
                    pl{count}= l;
                    count = count+1;
                end
            end
        else % direction postivie et > 1
            for j= 1-2*lss:3*lss
                l=zeros(2*lss);
                %l=[];
                % pour chaque pixel de cette droite
                for i=1:taillemax
                    X = 2*lss+b(i,2);
                    Y = j + b(i,1);
                    rr = (X-0.5-xcenter)^2 + (Y-0.5-ycenter)^2;
                    if (X>0 & X < 2*lss+1 && Y>0 & Y < 2*lss+1 & ( rr< r^2))
                        l(X,Y)=1;%l = [l ; X Y];
                    end
                end
                if sum(sum(l)) > 0
                    pl{count}= l;
                    count = count+1;
                end
            end
        end
    end
    pl = pl(~cellfun('isempty', pl));
end

function pl = genere_toutes_les_lignes(directions,N)
    [sx,~] = size(directions);
    lignes=cell(sx,1);
    pll=[];
    pl=[];
    A = [];
    for j=1:sx
        e10 = directions(j,2)/directions(j,1);
        A = [A; atan(e10)];
        pll = parallel_lines(e10,N);
        lignes{j}= pll;
    end
    [B,I] = sort(A);
    for j=1:sx
        r.d = B(j);
        r.lines = lignes{I(j)};
        pl = [pl; r];
    end
end


function pl = parallel_lines_l(direction,ls)
    r = sqrt(2)*ls/2;
    lss = ceil(r);
    xcenter = lss;
    ycenter = lss;
    taillemax = 4*lss;
    b = bresenham(direction,taillemax);
    pl=cell(taillemax,1);
    count=1;
    long_min_seg = ls/2;
    % ce qui suit partitionne bien le support si 
    % si -1 <= direction <= 1 
    
    if abs(direction) <= 1
        % pour chaque ligne
        for i= 1-2*lss:3*lss
            l=[];
            % pour chaque pixel de cette droite
            for j=1:taillemax
                X = i+b(j,2);
                Y = 1 + b(j,1) ;
                rr = (X-0.5-xcenter)^2 + (Y-0.5-ycenter)^2;
                if (X>0 & X < 2*lss+1 & Y>0 & Y < 2*lss+1 & ( rr< r^2))
                    l = [l ; X Y];
                end
            end
            if length(l) > long_min_seg
                pl{count}= l;
                count = count+1;
            end
        end
    else
        if direction <0
            for j= 1-2*lss:3*lss
                l=[];
                % pour chaque pixel de cette droite
                for i=1:taillemax
                    X = 1+b(i,2);
                    Y = j + b(i,1);
                    rr = (X-0.5-xcenter)^2 + (Y-0.5-ycenter)^2;
                    if (X>0 & X < 2*lss+1 & Y>0 & Y < 2*lss+1 & ( rr< r^2))
                        l = [l ; X Y];
                    end
                end
                if length(l) > long_min_seg
                    pl{count}= l;
                    count = count+1;
                end
            end
        else % direction postivie et > 1
            for j= 1-2*lss:3*lss
                l=[];
                % pour chaque pixel de cette droite
                for i=1:taillemax
                    X = 2*lss+b(i,2);
                    Y = j + b(i,1);
                    rr = (X-0.5-xcenter)^2 + (Y-0.5-ycenter)^2;
                    if (X>0 & X < 2*lss+1 & Y>0 & Y < 2*lss+1 & ( rr< r^2))
                        l = [l ; X Y];
                    end
                end
                if length(l) > long_min_seg
                    pl{count}= l;
                    count = count+1;
                end
            end
        end
    end
    
end


function  [directions,accuracy] = directions_of_pixels(cover,wp,params,level_delta,coverPadded)
    s= size(cover);
    sc=log2(s(1));
    directions=zeros(s);
    accuracy=zeros(s);
    decallageBloc = 2^(level_delta-1)
    for i=1:s(1)
        for j=1:s(2)
            [angle,aa] = recup_angle(cover,sc,wp,i-decallageBloc,j-decallageBloc,sc-level_delta,params,coverPadded);
            directions(i,j)=angle;
            accuracy(i,j) = aa;
        end
    end
end
   

function [angle,aa] = recup_angle(cover,sc,wp,i,j,niveau_decomp,params,coverPadded)
    angle = NaN;
    tailleBloc = 2^(sc-niveau_decomp);    
    blocInfo = params.blocInfos(sc-niveau_decomp);
    nbdir       = blocInfo.nbdir;
    

    % pour chaque niveau d'ondelette
    list_of_direction_proj_=cell(3,1);
    variances = [];
    for level=1:1 % TODO : update this !!!!!!!!!!!!!!
        list_of_direction_proj = [];
        blk=wp{level}(blocInfo.bloc_delta_m+i+1:i+tailleBloc+blocInfo.bloc_delta_p,blocInfo.bloc_delta_m+j+1:j+tailleBloc+blocInfo.bloc_delta_p);
        % pour chaque angle, calculer la projection
        for d=1:nbdir
            lpr = blocInfo.lines(d);
            direc = lpr.d ;
            nbl = length(lpr.lines);
            acc = 0;
            for l=1:nbl
                m = lpr.lines(l);
                p = blk(m{1}>0);
                delta_p = max(p)- min(p);
                acc = acc + delta_p;
            end
            acc = acc/nbl;
            list_of_direction_proj= [list_of_direction_proj; direc acc];
        end
        list_of_direction_proj_{level} = list_of_direction_proj;
        % maximiser la variance de la série
        stdd = std(list_of_direction_proj,0,2);
        variances = [variances ; stdd(2)];
    end    
    [vav,idx]= max(variances);
    projections_optimales= list_of_direction_proj_{idx};
%     subplot(2, 1, 1); imshow(uint8(coverPadded(blocInfo.bloc_delta_m+i+1:i+tailleBloc+blocInfo.bloc_delta_p,blocInfo.bloc_delta_m+j+1:j+tailleBloc+blocInfo.bloc_delta_p)));
%     subplot(2, 1, 2); plot(projections_optimales(1:nbdir),projections_optimales(nbdir+1:end));
%%%%%%%%%%%%% REVOIR CES CRITERES
    [mi,idx_min] = min(projections_optimales(:,2));
    angle = projections_optimales(idx_min,1);
    aa =angle_acurracy(projections_optimales,nbdir);
end

    
function acc=angle_acurracy(projections_optimales,nb)
    [max_vals,max_idx_angles]=findpeaks(projections_optimales(:,2));
    datainv = 1.01*max(projections_optimales(:,2))-projections_optimales(:,2);
    [min_vals,min_idx_angles]=findpeaks(datainv);
    acc=0;
    nbmin = length(min_idx_angles);
    nbmax = length(max_idx_angles);
    nb_iter= min(nbmin,nbmax);
    if (min(max_idx_angles)<min(min_idx_angles))
        start = max_idx_angles;
        next = min_idx_angles;
    else
        next = max_idx_angles;
        start = min_idx_angles;
    end
    
    if nbmin == nbmax   
        for j=1:nb_iter
            acc = acc+abs(projections_optimales(start(j),2)-projections_optimales(next(j),2));
        end
        for j=1:nb_iter-1
            acc = acc+ abs(projections_optimales(start(j+1),2)-projections_optimales(next(j),2));
        end
    else
        for j=1:nb_iter
            acc = acc+abs(projections_optimales(start(j),2)-projections_optimales(next(j),2));
        end
        for j=1:nb_iter
            acc = acc+abs(projections_optimales(start(j+1),2)-projections_optimales(next(j),2));
        end
    end
    amplitude = max(projections_optimales(:,2))-min(projections_optimales(:,2));
    if amplitude > 0 
        acc = acc / amplitude;
    else
        acc= NaN;
    end
end



function  [qtm,qtv] = qt_mydecomp(cover,wp,params)
    s= size(cover);
    sc=log2(s(1));
    maxdecomp = sc-1;
    [ilf,jlf,vlf] = qt_mydecomp_r(1,1,0);
    qtm = sparse(ilf,jlf,vlf(:,1),s(1),s(2));
    qtv = sparse(ilf,jlf,vlf(:,2),s(1),s(2));
    
    function [il,jl,vl] = qt_mydecomp_r(i,j,niveau_decomp)
        [split,angle] = a_decouper(cover,sc,wp, i,j,niveau_decomp,params);
        if ( split ==0 | niveau_decomp==maxdecomp)
            taillez = 2^(sc-niveau_decomp);
            il = [1+taillez*(i-1)]; jl = [1+taillez*(j-1)];
            vl = [taillez angle];
        else
            il = []; jl = []; vl = [];
            for ip=1:2
                for jp=1:2
                    [ir,jr,vr] = qt_mydecomp_r(2*(i-1)+ip,2*(j-1)+jp,niveau_decomp+1);
                    il = [il ; ir]; jl = [jl ; jr]; vl = [vl ; vr];
                end
            end
        end
    end
end



function [split,angle] = a_decouper(cover,sc,wp,i,j,niveau_decomp,params)
    angle = NaN;
    split = 1;
    tailleBloc = 2^(sc-niveau_decomp);
    if tailleBloc>16
        return
    end    
    
    blocInfo = params.blocInfos(sc-niveau_decomp);
    nbdir       = blocInfo.nbdir;
    

    % pour chaque niveau d'ondelette
    list_of_direction_proj_=cell(3,1);
    variances = [];
    for level=1:3 
        list_of_direction_proj = [];
        blk=wp{level}(blocInfo.bloc_delta_m+(i-1)*tailleBloc+1:i*tailleBloc+blocInfo.bloc_delta_p,blocInfo.bloc_delta_m+(j-1)*tailleBloc+1:j*tailleBloc+blocInfo.bloc_delta_p);
        % pour chaque angle, calculer la projection
        for d=1:nbdir
            lpr = blocInfo.lines(d);
            direc = lpr.d ;
            nbl = length(lpr.lines);
            acc = 0;
            for l=1:nbl
                m = lpr.lines(l);
                p = blk(m{1}>0);
                delta_p = max(p)- min(p);
                acc = acc + delta_p;
            end
            acc = acc/nbl;
            list_of_direction_proj= [list_of_direction_proj; direc acc];
        end
        list_of_direction_proj_{level} = list_of_direction_proj;
        % maximiser la variance de la série
        stdd = std(list_of_direction_proj,0,2);
        variances = [variances ; stdd(2)];
    end    
    [vav,idx]= max(variances);
    projections_optimales= list_of_direction_proj_{idx};
%     % ne pas découpler si signal de très faible amplitude 
%    subplot(2, 1, 1); imshow(uint8(cover((i-1)*tailleBloc+1:i*tailleBloc,(j-1)*tailleBloc+1:j*tailleBloc)));
%    subplot(2, 1, 2); plot(projections_optimales(1:nbdir),projections_optimales(nbdir+1:end));
%%%%%%%%%%%%% REVOIR CES CRITERES
    [split,angle] = decision_de_decouper(projections_optimales,nbdir);
end


function [split,angle] = decision_de_decouper(projections_optimales,nbdir)
    split = 1;
    [mi,idx_min] = min(projections_optimales(:,2));
    angle = projections_optimales(idx_min,1);
    % cf Reeth thesis p 77.
    Vref = (projections_optimales(nbdir/2,2)+projections_optimales(nbdir,2))/2;
    if (mi <= 0.5*Vref)
        split = 0;
    end
end


function [split,angle] = decision_de_decouper_jf(projections_optimales,vav)
    [mx,idx_max] = max(projections_optimales(1:end,2:end));
    [mi,idx_min] = min(projections_optimales(1:end,2:end));
    split = 1;
    if (vav < 10)
        split = 0;
        angle = projections_optimales(idx_min,1);
    else
        tp=0.17;
        intervalle_mx = mi+sqrt(vav);
        if length(find(projections_optimales(:,2)<intervalle_mx))/nbdir < tp
            split = 0;
            angle = projections_optimales(idx_min,1);
        end
    end
end


function display_qt(cover,smq)
    I = uint8(cover);
    S = smq;
    blocks = repmat(uint8(0),size(S));

    for dim = [512 256 128 64 32 16 8 4 2 1];
        numblocks = length(find(S==dim));
        if (numblocks > 0)
            values = repmat(uint8(1),[dim dim numblocks]);
            values(2:dim,2:dim,:) = 0;
            blocks = qtsetblk(blocks,S,dim,values);
        end
    end

    blocks(end,1:end) = 1;
    blocks(1:end,end) = 1;

    imshow(I), figure, imshow(blocks,[]);
end



function v = variation_de_direction(spv,fs,rad)
    [spvl,spvc] = size(spv);
    spvPadded = padarray(spv, [fs fs], 'symmetric');
    v = zeros(spvl,spvc);
    for i=1:spvl
        for j=1:spvc
            acc = 0;
            for ip=0:2*fs
                for jp=0:2*fs
                   acc = acc + dist(spv(i,j),spvPadded(i+ip,j+jp),rad);
                end
            end
            v(i,j)=acc;
        end
    end
end

function v = variation_d_amplitude(spv,fs)
    [spvl,spvc] = size(spv);
    spvPadded = padarray(spv, [fs fs], 'symmetric');
    v = zeros(spvl,spvc);
    for i=1:spvl
        for j=1:spvc
            acc = 0;
            for ip=0:2*fs
                for jp=0:2*fs
                   acc = acc + abs(spv(i,j)-spvPadded(i+ip,j+jp));
                end
            end
            v(i,j)=acc;
        end
    end
end






function d = dist(a1,a2,rad)
    d2 = abs(a2-a1);
    if rad == true
        d= min(pi-d2,d2);    
    else
        d=min(360-d2,d2);
    end
end
    
function fullM=enUneMatriceFull(spm,spv)
    sps= size(spm);
    fullM = zeros(sps);
    for dim = [512 256 128 64 32 16 8 4 2];    
        blcs = find(spm==dim);
        for idx = 1:length(blcs); 
            [I,J] = ind2sub(sps,blcs(idx,1));
            v = spv(I,J);
            fullM(I:I+dim-1,J:J+dim-1)=v;
        end
    end
end

function Ap = amplitude_extremes(A)
    [nbl,nbc] = size(A);
    un = ones(3,3);
    Ap=zeros(nbl,nbc);
    T=2;
    for i=2:nbl-1
        for j=2:nbc-1
            k = A(i-1:i+1,j-1:j+1)-(A(i,j)*un);
            if sum(sum(k>T)) == 8
                Ap(i,j)= exp(sum(sum(k)));
            else
            if sum(sum(k<-T)) == 8
                Ap(i,j)= exp(abs(sum(sum(k))));
            end
            end
        end
    end

end
