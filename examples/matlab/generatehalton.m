clc
clear all
close all

diary('matlaboutput.txt')

samps = [50 74 110 164 244 362 538 800];
runs = 2000;
dim = 10;

for i = 1:length(samps)
    nsamp = samps(i);
    for j = 1:runs
        skip = randi([0,1e6]);
        leap = 100;
        p = haltonset(dim,'Skip', skip,'Leap',leap);
        p = scramble(p, 'RR2');
        p(1:nsamp,:)
    end
end