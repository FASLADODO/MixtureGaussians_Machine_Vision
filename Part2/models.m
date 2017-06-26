load('RGBAppleTrain', 'appleTrain');
load('RGBNonAppleTrain','nonAppleTrain');

trainModels(1,appleTrain,nonAppleTrain);
gmApple3FitMixGauss = fitMixGauss(appleTrain,3);
gmNonApple3FitMixGauss = fitMixGauss(nonAppleTrain,3);
save('gmApple3FitMixGauss.mat','gmApple3FitMixGauss');
save('gmNonApple3FitMixGauss.mat','gmNonApple3FitMixGauss')

% if flag set to 1 then built in function fitgmdist will be used to obtain
% the mixture of Gaussians, else the fitMixGauss function developed in part
% c) will be used
function trainModels(flag,appleTrain,nonAppleTrain)
    if flag == 1
        gmApple1 = fitgmdist(appleTrain',1);
        gmApple2 = fitgmdist(appleTrain',2);
        gmApple3 = fitgmdist(appleTrain',3);
        gmApple4 = fitgmdist(appleTrain',4);

        gmNonApple1 = fitgmdist(nonAppleTrain',1);
        gmNonApple2 = fitgmdist(nonAppleTrain',2);
        gmNonApple3 = fitgmdist(nonAppleTrain',3);
        gmNonApple4 = fitgmdist(nonAppleTrain',4);
    else 
        gmApple1 = fitMixGauss(appleTrain',1);
        gmApple2 = fitMixGauss(appleTrain',2);
        gmApple3 = fitMixGauss(appleTrain',3);
        gmApple4 = fitMixGauss(appleTrain',4);

        gmNonApple1 = fitMixGauss(nonAppleTrain',1);
        gmNonApple2 = fitMixGauss(nonAppleTrain',2);
        gmNonApple3 = fitMixGauss(nonAppleTrain',3);
        gmNonApple4 = fitMixGauss(nonAppleTrain',4);
    end;
    gmApple = {gmApple1,gmApple2,gmApple3,gmApple4};
    gmNonApple = {gmNonApple1,gmNonApple2,gmNonApple3,gmNonApple4};
    save('gmApple.mat','gmApple');
    save('gmNonApple.mat','gmNonApple')
end


