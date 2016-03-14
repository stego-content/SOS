function [stego, distortion,rho] = poly_n(cover_, payload,params,kl)
    cover = double(cover_);
    sizeCover=size(cover);
    p=params.p;
    nn=params.n;
    wetCost = 10^10;
 
    
    
    d2p_dx2_o = zeros(sizeCover);
    d2p_dy2_o= zeros(sizeCover);
    d2p_dy_dx_o = zeros(sizeCover);    
 
    
    for n=1:nn
        padSize = n;
        coverPadded = padarray(cover, [padSize padSize], 'symmetric');

        k = kl{n}.h;
        d2p_dx2= conv2(coverPadded,k, 'same');    
        Ap = d2p_dx2;
        d2p_dx2 = Ap(((size(Ap, 1)-sizeCover(1))/2)+1:end-((size(Ap, 1)-sizeCover(1))/2), ((size(Ap, 2)-sizeCover(2))/2)+1:end-((size(Ap, 2)-sizeCover(2))/2));

    
        d2p_dy2= conv2(coverPadded,k', 'same');
        Ap = d2p_dy2;
        d2p_dy2 = Ap(((size(Ap, 1)-sizeCover(1))/2)+1:end-((size(Ap, 1)-sizeCover(1))/2), ((size(Ap, 2)-sizeCover(2))/2)+1:end-((size(Ap, 2)-sizeCover(2))/2));

        k = kl{n}.d;
        d2p_dy_dx= conv2(coverPadded,k, 'same');
        Ap = d2p_dy_dx;
        d2p_dy_dx = Ap(((size(Ap, 1)-sizeCover(1))/2)+1:end-((size(Ap, 1)-sizeCover(1))/2), ((size(Ap, 2)-sizeCover(2))/2)+1:end-((size(Ap, 2)-sizeCover(2))/2));

        d2p_dx2_o = max(abs(d2p_dx2),d2p_dx2_o);
        d2p_dy2_o = max(abs(d2p_dy2),d2p_dy2_o);
        d2p_dy_dx_o = max(abs(d2p_dy_dx),d2p_dy_dx_o);
    end
    

    
    rho= ((d2p_dx2_o).^(p) + (d2p_dy2_o).^(p) + ((d2p_dy_dx_o)).^(p)).^(-1/p);

    
    

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
