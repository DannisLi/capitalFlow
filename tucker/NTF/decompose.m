function [] = decompose(coreNway,lam)

load whole_month_log M;

M = tensor(M);
opts = [];
opts.maxit = 2000;
opts.lam = lam;
opts.tol = 2e-7;

[A,C,out] = tucker(M, coreNway, opts);

O = A{1};
D = A{2};
T = A{3};
C = C.data;


save result2 O D T C;

