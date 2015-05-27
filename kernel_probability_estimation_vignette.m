%%
% Simulation 1 Rate Generator - Step And Ramp
sessions = 30;
kx = (0:300)';
simrate = 0.2.*ones(301,1);
simrate(40:90) = 0.7;
simrate(180:280) = 0.8-((0:0.005:0.5)./0.5)*0.6;
samp_size = [32;96;301];

%%
% Simulation 2 Rate Generator - Single Step
sessions = 30;
kx = (0:300)';
simrate = 0.1.*ones(301,1);
simrate(150:end) = 0.9;

%%
% Simulation 3 Rate Generator- Sine Wave
sessions = 30;
kx = (0:300)';
simrate = sin(kx./(6*pi))./3 + 0.5;

%%
% Simulation 4 - Logarithmic Learning
sessions = 30;
kx = (0:300)';
simrate = 1./(1+exp(-0.015.*kx));

%%
% Simulator

q = cell(2,3);

for j = 1:3
	y_t = [];
	X_t = [];
	for i = 1:sessions
		rdex = sort(randsample(301,samp_size(j)));
		y_t = [y_t;binornd(1,simrate(rdex))];
		X_t = [X_t;kx(rdex)];
	end
	[q{1,j},~,q{2,j}] = kernel_probability_estimate(y_t,X_t,sessions,kx);
	plot(kx,simrate,kx,q{1,j},kx,q{2,j})
end

%%
% Multinomial Demonstration

D = load('rat_transition.csv');

Y = [D(:,2)==2 D(:,2)==3 D(:,2)==4 D(:,2)==5] + 0;
X = D(:,5);
sessions = length(unique(D(:,1)));
span = 0.01;
q = cell(2,2);

%before
[q{1,1},~,~,q{2,1}] = kernel_probability_estimate(Y(X<=0,:),X(X<=0),sessions,(-1500:0)',span);
[q{1,2},~,~,q{2,2}] = kernel_probability_estimate(Y(X>=0,:),X(X>=0),sessions,(0:1500)',span);
plot((-1500:0),q{1,1},(0:1500),q{1,2});