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
% Binomial Demonstration - Transitive Inference

D = load('monkey_ti.csv');
% Eliminate terminal item pairs
dex = find((D(:,6)==1)|(D(:,7)==7));
D(dex,:) = [];

y = D(:,1);
X = D(:,4);
span = 1;

q_all = cell(1,4);
ci_all= cell(1,4);
cnt_all= cell(1,4);
sessions = length(unique(D(:,3)));
% Get kernel probability estimates
hold on
for d = 1:4
	if d == 1
		kx = (-200:400)';
	else
		kx = (0:400)';
	end
	dex = find(D(:,5)==d);
	[q,kx,bw,ci,cnt] = kernel_probability_estimate(y(dex),X(dex),sessions,kx,span);
	q_all{1,d}  = [kx q];
	ci_all{1,d} = [kx ci(:,1) kx ci(:,2)];
	cnt_all{1,d}= cnt;
	plot(kx,q)
end
hold off

%%
% Multinomial Demonstration - Rat Choice

D = load('rat_choice.csv')';

Y = [D==0 D==1 D==2 D==3 D==4 D==5 D==6 D==7] + 0;
X = (1:size(D,1))';
phase = [0;4624;9756;14303;18692;23907;28111;33142;38164];
span = 1;
sched = [
	0.0422,0.0357,0.0617,0.0552,0.0097,0.0065,0.0227,0.0162;
	0.0552,0.0162,0.0357,0.0065,0.0617,0.0227,0.0097,0.0442;
	0.0162,0.0617,0.0097,0.0357,0.0227,0.0552,0.0422,0.0065;
	0.0617,0.0422,0.0065,0.0162,0.0552,0.0097,0.0357,0.0227;
	0.0065,0.0097,0.0162,0.0227,0.0357,0.0422,0.0552,0.0617;
	0.0357,0.0227,0.0552,0.0617,0.0422,0.0162,0.0065,0.0097;
	0.0227,0.0065,0.0422,0.0097,0.0162,0.0357,0.0617,0.0552;
	0.0097,0.0552,0.0224,0.0422,0.0065,0.0617,0.0162,0.0357];

% Use kernel probability estimation for instantaneous estimation at all
% time points.
for i = 1:length(phase)-1
	kx = (phase(i)+1:phase(i+1))';
	[q,kx,bw,ci,cnt] = kernel_probability_estimate(Y(kx,:),kx,1,kx,1);
	qest(kx,:) = q;
end

subplot(3,1,1)
plot(X,qest);

% Factor out bias and perform centered log-ratio transformation
bias = geomean(qest);
qmod = (qest./repmat(bias,38164,1));
qmod = log(qmod./repmat(geomean(qmod,2),1,8));

subplot(3,1,2)
plot(X,qmod)

% Estimate sensitivity
creg = zeros(38164,1);
warning('off','stats:statrobustfit:IterationLimit')
for i = 1:length(phase)-1
kx = (phase(i)+1:phase(i+1))';
for j = 1:length(kx)
creg(kx(j)) = robustfit(csched(i,:)',cqmod(kx(j),:)',[],[],'off');
end
end
warning('on','stats:statrobustfit:IterationLimit')

subplot(3,1,3)
plot(X,creg)
