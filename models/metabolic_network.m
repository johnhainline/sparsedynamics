% older model
% https://www.ebi.ac.uk/biomodels/MODEL1008120002#Overview

% docs https://www.mathworks.com/help/simbio/ref/sbiosimulate.html

% model https://www.ebi.ac.uk/biomodels-main/BIOMD0000000619
model = sbmlimport('data/BIOMD0000000619.xml');

% default config
config = getconfigset(model,'default');
config.TimeUnits = 'hour';
config.StopTime = 25;

% add a dose of Acetaminophen (APAP)
%dose1 = adddose(model,'d1','schedule');
%dose1.Amount = 1.4;
%dose1.AmountUnits = 'gram';
%dose1.TimeUnits = 'second';
%dose1.Time = 1;
%dose1.TargetName = 'VVen.CVen';

% run simulation
%[t,x,names] = sbiosimulate(model,config,dose1);
[t,x,names] = sbiosimulate(model,config);

% plot the amount of APAP in the veins over time
figure(1);
venousAPAPInLiters = x(:, 6) / 3.41;
plot(t,venousAPAPInLiters);
xlabel('Time (hrs)');
ylabel('Venous APAP (mol/L)');
title('Metabolism of Acetaminophen (APAP)');
% avoid exponential notation
ax = gca;
ax.YAxis.Exponent = 0;

%figure(2);
%plot(t,x(:, 11:17));
%xlabel('Time');
%ylabel('States');
%title('States vs Time');
%legend(names(11:end));
