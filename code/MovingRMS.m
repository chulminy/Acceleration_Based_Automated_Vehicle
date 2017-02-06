function y = MovingRMS (signal, windowlength)

% CALCULATE RMS
y = zeros(length(signal),1);

signal = [zeros(windowlength/2,1);signal;zeros(windowlength/2,1)];

% Square the samples
signal = signal.^2;

index = 0;
for i = windowlength/2:length(signal)-windowlength/2-1
	index = index+1;
	% Average and take the square root of each window
	y(index) = sqrt(mean((signal(i-windowlength/2+1:i+windowlength/2))));
end
