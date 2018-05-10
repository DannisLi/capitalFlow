function result = tucker(M,coreNway,opts)
%function A = ntds_fapg(M,coreNway,opts)
% ntds: sparse nonnegative Tucker decomposition
%  min 0.5*||M - C \times_1 A_1 ...\times_N A_N||_F^2 
%   + sum_{n=1}^{N} lam_n*||A_n||_1 + lam_{N+1}*||C||_1 
%  subject to A_1>=0, ..., A_N>=0, C>=0
% input:
%       M: input nonnegative tensor
%       coreNway: size of core tensor C
%       opts.
%           tol: tolerance for relative change of function value, default:
%           1e-4
%           maxit: max number of iterations, default: 500
%           maxT: max running time, default: 1e3
%           rw: control the extrapolation weight, default: 1
%           A0: initial point in cell struct, default: Gaussian random
%           matrices
%           C0: initial value of C, default: Gaussian random array
% output:
%       A: cell struct with each component being nonnegative matrix
%       C: nonnegative core tensor
%       Out.
%           iter: number of iterations
%           hist_obj: history of objective values at each iteration
%           hist_rel: history of relative changes at each iteration
%
% require the Toolbox of tensor
% downloaded from http://www.sandia.gov/~tgkolda/TensorToolbox/
%
% More information can be found at:
% http://www.caam.rice.edu/~optimization/bcu/

%% Parameters and defaults
if isfield(opts,'maxit')   maxit = opts.maxit;     else maxit = 500;   end
if isfield(opts,'tol')     tol = opts.tol;         else tol = 1e-4;    end
if isfield(opts,'rw')      rw = opts.rw;           else rw = 0.9999;   end
if isfield(opts,'maxT')    maxT = opts.maxT;       else maxT = 1e3;    end
if isfield(opts,'hosvd')   hosvd = opts.hosvd;     else hosvd = 0;     end
if isfield(opts,'Lmin')    Lmin = opts.Lmin;       else Lmin = 1;      end

%% Data preprocessing and initialization
M = tensor(M);
N = ndims(M); % M is an N-way tensor
Nway = M.size; % dimension of M

if isfield(opts,'A0')
    A0 = opts.A0;
else
    A0 = cell(1,N);
    for n = 1:N
        % randomly generate each factor matrix
        A0{n} = max(0,randn(Nway(n),coreNway(n)));
    end
end

if isfield(opts,'C0')
    C0 = tensor(opts.C0);
else
    % randomly generate core tensor
    C0 = tensor(max(0,randn(coreNway)));
end

% pre-process the starting point
if hosvd
    for n = 1:N
        Atilde = ttm(M, A0, -n, 't');
        A0{n} = max(eps,nvecs(Atilde,n,coreNway(n)));
    end
    A0 = cellfun(@(x) bsxfun(@rdivide,x,sum(x)),A0,'uni',0);
    C0 = ttm(M, A0, 't');
end

% coefficient of l_1 regularization terms
% lam(N+1) for core tensor C; 
% lam(n) for factor matrix A{n}, n = 1,...,N
if isfield(opts,'lam') lam = opts.lam; else lam = zeros(1,N+1); end

% check existence of sparseness regularizer
if max(lam)>0 sp = 1; else sp = 0; end

% add bound constraint for well-definedness
doproj = zeros(1,N+1); 
doproj(lam==0) = 1; tau = inf;
if isfield(opts,'bound') tau = opts.bound; end

Mnrm = norm(M); 
Asq = cell(1,N); nrmA = zeros(1,N);

% rescale the initial point according to number of elements
Nnum = Nway.*coreNway;
totalNum = prod(coreNway)+sum(Nnum);
for n = 1:N   
    A0{n} = A0{n}/norm(A0{n},'fro')*Mnrm^(Nnum(n)/totalNum);
    Asq{n} = A0{n}'*A0{n};
    nrmA(n) = norm(Asq{n});
end
C0 = tensor(C0/norm(C0)*Mnrm^(prod(coreNway)/totalNum));

obj0 = 0.5*Mnrm^2;
if sp 
    obj0 = obj0+lam(N+1)*sum(C0.data(:)); 
    for n = 1:N
        obj0 = obj0+lam(n)*sum(A0{n}(:));
    end
end
obj = obj0;

A = A0; Am = A0; Asq0 = Asq;
C = C0; Cm = C0;

nstall = 0; w = 0; t0 = ones(N+1,1); t = t0; wA = zeros(N+1,1);
L0 = ones(N+1,1); L = ones(N+1,1); Out.redoN = 0; 

%% Store data to save computing time if it is not too large
storedata = false;
if N*prod(Nway)<4000^2
    storedata = true;
    pM = cell(1,N);
    for n = 1:N
        pM{n} = tenmat(M,n); pM{n} = pM{n}.data;
    end
end
%% Iterations of block-coordinate update
%
%  iteratively updated variables:
%       GradA: gradients with respect to each component matrix of A
%       GradC: gradient with respect to C
%       A,C: new updates
%       A0,C0: old updates
%       Am,Cm: extrapolations of A
%       L, L0: current and previous Lipschitz bounds
%       obj, obj0: current and previous objective values

start_time = tic;

for k = 1:maxit
    objn0 = obj;
    MtA0{1} = ttm(M,A0{1},1,'t');
    for i = 2:N
        MtA0{i} = ttm(MtA0{i-1},A0{i},i,'t');
    end
    for n = N:-1:1
        % -- update the core tensor C --
        L0(N+1) = L(N+1);
        L(N+1) = 1;
        for i = 1:N L(N+1) = L(N+1)*nrmA(i); end
        L(N+1) = max(Lmin,L(N+1));
        CtA = ttm(Cm, Asq);
        MtA = MtA0{n};
        for i = n+1:N
            MtA = ttm(MtA,A{i},i,'t');
        end
        % compute the gradient
        GradC = CtA.data-MtA.data;
        if sp
            Cdata = max(0,Cm.data-GradC/L(N+1)-lam(N+1)/L(N+1));
        else
            Cdata = max(0,Cm.data-GradC/L(N+1));
        end
        
        if doproj(end)
            % do projection
            Cdata(Cdata>tau) = tau;
        end
        C = tensor(Cdata);
        % -- update n-th factor matrice --
        if storedata
            B = tenmat(ttm(C,A,-n),n);
            Bsq = B.data*B.data';
            MB = pM{n}*B.data';
        else
            B = tenmat(ttm(C,Asq,-n),n);
            MB = tenmat(ttm(M,A,-n,'t'),n);
            tCn = tenmat(C,n)';
            
            Bsq = B*tCn; Bsq = Bsq.data;
            MB = MB*tCn; MB = MB.data;            
        end
        %compute the gradient
        GradA = Am{n}*Bsq-MB;
        L0(n) = L(n);
        L(n) = norm(Bsq);
        L(n) = max(Lmin,L(n));
        if sp
            A{n} = max(0,Am{n}-GradA/L(n)-lam(n)/L(n));
        else
            A{n} = max(0,Am{n}-GradA/L(n));
        end
        
        if doproj(n)
            % do projection
            A{n}(A{n}>tau) = tau;
        end

        
        Asq{n} = A{n}'*A{n}; nrmA(n) = norm(Asq{n});
        res = 0.5*(sum(sum(Asq{n}.*Bsq))-2*sum(sum(A{n}.*MB))+Mnrm^2);
        objn = res;
        if sp
            objn = objn+lam(N+1)*sum(C.data(:));
            for i = 1:N
                objn = objn+lam(i)*sum(A{i}(:));
            end
        end
        
        if objn>objn0
            Out.redoN = Out.redoN+1;
            % re-update to make objective nonincreasing 
            Asq{n} = Asq0{n};
            CtA = ttm(C0, Asq);  
            GradC = CtA.data-MtA.data; % compute the gradient
            if sp
                Cdata = max(0,C0.data-GradC/L(N+1)-lam(N+1)/L(N+1));
            else
                Cdata = max(0,C0.data-GradC/L(N+1));
            end

            if doproj(end)
                % do projection
                Cdata(Cdata>tau) = tau;
            end
            C = tensor(Cdata);
            
            if storedata
                B = tenmat(ttm(C,A,-n),n);
                Bsq = B.data*B.data';
                MB = pM{n}*B.data';
            else
                B = tenmat(ttm(C,Asq,-n),n);
                MB = tenmat(ttm(M,A,-n,'t'),n);
                tCn = tenmat(C,n)';

                Bsq = B*tCn; Bsq = Bsq.data;
                MB = MB*tCn; MB = MB.data;            
            end
            
            GradA = A0{n}*Bsq-MB; % compute the gradient
            L(n) = norm(Bsq);
            L(n) = max(Lmin,L(n));
            if sp
                A{n} = max(0,A0{n}-GradA/L(n)-lam(n)/L(n));
            else
                A{n} = max(0,A0{n}-GradA/L(n));
            end
            
            if doproj(n)
                % do projection
                A{n}(A{n}>tau) = tau;
            end
        
            Asq{n} = A{n}'*A{n}; nrmA(n) = norm(Asq{n});
            res = 0.5*(sum(sum(Asq{n}.*Bsq))-2*sum(sum(A{n}.*MB))+Mnrm^2);
            objn = res;
            if sp
                objn = objn+lam(N+1)*sum(C.data(:));
                for i = 1:N
                    objn = objn+lam(i)*sum(A{i}(:));
                end
            end
        end
        
        % do extrapolation
        t(n) = (1+sqrt(1+4*t0(n)^2))/2;
        w = (t0(n)-1)/t(n); % extrapolation weight
        % choose smaller weight for convergence
        wA(n) = min([w,rw*sqrt(L0(n)/L(n))]); 
        Am{n} = A{n}+wA(n)*(A{n}-A0{n});
        t(N+1) = (1+sqrt(1+4*t0(N+1)^2))/2;
        w = (t0(N+1)-1)/t(N+1); % extrapolation weight
        % choose smaller weight for convergence
        wA(N+1) = min([w,rw*sqrt(L0(N+1)/L(N+1))]);
        Cm = tensor(C.data+wA(N+1)*(C.data-C0.data));
        
        % store old update
        A0{n} = A{n}; C0 = C; objn0 = objn; Asq0{n} = Asq{n};
        t0(n) = t(n); t0(N+1) = t(N+1);
    end
    % --- diagnostics, reporting, stopping checks ---
    obj0 = obj; obj = objn;
    relerr1 = abs(obj-obj0)/(obj0+1);    relerr2 = (2*res)^.5/Mnrm;
    
    
    % check stopping criterion
    crit = relerr1<tol;
    if crit; nstall = nstall+1; else nstall = 0; end
    if nstall>=3 || relerr2<tol break; end
    if toc(start_time)>maxT; break; end;
result = A;
result{N+1} = C.data;
end