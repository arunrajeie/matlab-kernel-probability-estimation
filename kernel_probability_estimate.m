function [prob, kx, bw, ci, count] = kernel_probability_estimate(Y,X,n,kx,span)
%KERNEL_PROBABILITY_ESTIMATE Yields a kernel probability estimate based on a categorical time series.
%
%==Input Parameters==
%
%	Y		Categorial outcomes, coded as a presence (1) or absence (0) of a category. If Y
%			has one column, it is presumed to report successes in a binomial time series.
%	X		Recorded events times. X and Y are expected to have the same number of rows.
%	n		The number of independently samples time series. Default == 1.
%	kx		Target times for which estimates are desired. Need not align with the times in
%			X, or consist of integers. Default == min(X):1:max(X).
%	span	Minimum resolution between observations. Used in a correction for continuity.
%			Default == 1.
%
%==Output Parameters==
%
%	prob	Probability estimates for each category in Y at every time in kx.
%	kx		Set of times over which probabilities are to be estimated.
%	bw		Optimal bandwidths identified by the algorithm
%	ci		Credible intervals for the estimates in prob.
%	count	Absolute values recovered from each kernel, as well as the estimate of the
%			number of observations contributing to each estimate, for use in credible
%			interval construction.
%
%===References===
%
%	Brown LD, Cai TT, DasGupta A (2001) Interval estimation for a binomial proportion.
%		Statistical Science, 16, 171-182.
%	Jensen G (Submitted) Kernel probability estimation for binomial and multinomial data.
%	Shimazaki H, Shinomoto S (2010) Kernel bandwidth optimization in spike rate
%		estimation. Journal of Computational Neuroscience, 29, 171-182.
%
% written by:
% Greg Jensen
% greg.guichard.jensen@gmail.com

% Infer minimum resolution of observation from data if span is not provided
if nargin < 5
	dfs = diff(unique(X));
	dfs(dfs==0) = [];
	span = min(dfs);
end

% Infer time interval if kx is not provided
if nargin < 4
	kx = (min(X):max(X))';
end

% If n is not reported, only one times series is assumed to exist in the data 
if nargin < 3
	n = 1;
end

len = length(kx);
ops = size(Y,2);
ot = ops;
if ops == 1
	Y = [Y 1-Y];
	ops = 2;
end
count = zeros(len,ops+1); %count(:,ops+1 is for total count)
prob = zeros(len,ops);
ci = zeros(len,ops,2);
bw = zeros(ops,1);

% Perform rate estimation
for c = 1:ops
	o_c = find(Y(:,c)==1);
	Xc = X(o_c,1);
	if length(Xc)>1
		fun = @(a) local_gaussian_cost_function(a,Xc,n,span);
		bw(c) = fminsearch(fun,100);
		for i = 1:length(Xc)
			d = normpdf(kx,Xc(i),bw(c));
			count(:,c) = count(:,c) + d;
			count(:,ops+1) = count(:,ops+1) + d.*bw(c).*sqrt(2*pi());
		end
	else
		count(:,c) = count(:,c) + 1./length(kx);
		bw(c) = (max(kx)-min(kx))./sqrt(12);
	end
end

% Probability and credible interval calculation, using the Jeffreys prior
if ot==1
	prob = count(:,1)./sum(count(:,1:ops),2);
	ci(:,1) = betainv(.975,(prob.*count(:,ops+1))+0.5,((1-prob).*count(:,ops+1))+0.5);
	ci(:,2) = betainv(.025,(prob.*count(:,ops+1))+0.5,((1-prob).*count(:,ops+1))+0.5);
else
	for c = 1:ops
		prob(:,c) = count(:,c)./sum(count(:,1:ops),2);
		ci(:,c,1) = betainv(.975,(prob(:,c).*count(:,ops+1))+0.5,((1-prob(:,c)).*count(:,ops+1))+0.5);
		ci(:,c,2) = betainv(.025,(prob(:,c).*count(:,ops+1))+0.5,((1-prob(:,c)).*count(:,ops+1))+0.5);
	end
end

end

function cost = local_gaussian_cost_function(w,d,n,gap)
%	Returns the Gaussian bandwidth cost metric described by Shimazaki & Shinomoto (2010)
	N = length(d);
	cost = 0;
	for i = 1:(length(d)-1)
		for j = (i+1):length(d)
			df = (d(i)-d(j))^2 + (gap^2)/2;
			cost = cost + exp((-df)/(4*w^2)) - 2*sqrt(2)*exp((-df)/(2*w^2));
		end
	end
	cost = ((2/w)*cost + N/w)/(2*sqrt(pi())*n^2);
end


