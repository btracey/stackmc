clc
clear all
close all

nSamples = [40 61.36509618537565 94.1418757460101 144.42563150563976 221.56731922443 339.9124963935743 521.4690758950713 799.9999999999999];
mcEse = [208242.1031654711 133585.26156328808 79111.18997810346 56169.91656785763 38170.84372292873 23690.72873572781 15972.934662111893 9474.74065646573];
mcEim = [7305.817100729906 4242.640604544245 2549.041724630441 1743.0474956125931 1182.3274878705547 750.781398759121 530.6792195451641 287.71478969267685];
fitEse = [203378.8165357968 132050.18058356654 78081.020039447 56255.20094841222 38119.01958066936 23649.985253321483 15980.631196670654 9473.349489347982];
fitEim = [6917.801763706064 4195.566201556408 2512.5406663849653 1744.2072244332983 1183.200749074505 749.2304315213356 530.5111241682708 287.705923189832];
stackMCEse = [347035.9099391628 224656.05344014431 108063.14745427825 65508.86333913394 42678.30493170218 24752.55273017943 16486.210053260842 9766.783955576038];
stackMCEim = [12436.897690309956 6975.398153849134 3562.6428870064587 2111.0602431935517 1328.2612203141873 793.3598382579758 553.6602773357534 297.12974610583734];
lhcEse = [401401.0505745774 151894.33938810212 95438.527753207 60130.63723752681 35721.06495651887 24245.57620284954 15478.863254105663 10170.439338116114];
lhcEim = [13018.239632156243 4836.389031568989 3039.975519153802 1843.7443844513602 1132.4745858290341 751.6004564822388 499.3193408840918 336.137524914034];
indEse = [1.6073089559656437e+06 413510.77977993776 197707.80390169666 102010.51545655484 64846.33206123711 37812.96462369385 22897.073735158374 14964.875098080054];
indEim = [51815.92890046743 13294.254856563462 6272.249011419187 3275.2322716818585 1963.2539220297701 1197.4151002413364 699.3228892203939 439.68962080273724];

labelFontSize = 16;
lineWidth = 2;
ax = [30 1000 5000 3e6];

figure
hold all
errorbar(nSamples, mcEse, mcEim, 'g','LineWidth', lineWidth)
errorbar(nSamples, fitEse,fitEim, 'r','LineWidth', lineWidth)
errorbar(nSamples, stackMCEse, stackMCEim, 'b','LineWidth', lineWidth)
legend('MC','Polynomial','StackMC')
set(gca, 'XScale','log')
set(gca, 'YScale','log')
set(gca,'FontSize',14)
xlabel('Number of Samples','FontSize',labelFontSize)
ylabel('Expected Squared Error','FontSize',labelFontSize)
axis(ax)

figure
hold all
errorbar(nSamples, mcEse, mcEim, 'g','LineWidth', lineWidth)
errorbar(nSamples, fitEse,fitEim, 'r','LineWidth', lineWidth)
errorbar(nSamples, stackMCEse, stackMCEim, 'b','LineWidth', lineWidth)
errorbar(nSamples, indEse, indEim,'c','LineWidth', lineWidth)
legend('MC','Polynomial','StackMC','Independent Correction')
set(gca, 'XScale','log')
set(gca, 'YScale','log')
set(gca,'FontSize',14)
xlabel('Number of Samples','FontSize',labelFontSize)
ylabel('Expected Squared Error','FontSize',labelFontSize)
axis(ax)

figure
hold all
errorbar(nSamples, mcEse, mcEim, 'g','LineWidth', lineWidth)
errorbar(nSamples, fitEse,fitEim, 'r','LineWidth', lineWidth)
errorbar(nSamples, stackMCEse, stackMCEim, 'b','LineWidth', lineWidth)
errorbar(nSamples, lhcEse, lhcEim,'c','LineWidth', lineWidth)
legend('MC','Polynomial','StackMC','Latin Hypercube Correction')
set(gca, 'XScale','log')
set(gca, 'YScale','log')
set(gca,'FontSize',14)
xlabel('Number of Samples','FontSize',labelFontSize)
ylabel('Expected Squared Error','FontSize',labelFontSize)
axis(ax)
