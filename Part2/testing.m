close all;

load('RGBAppleTrain', 'appleTrain');
load('RGBNonAppleTrain','nonAppleTrain');
load('gmApple', 'gmApple');
load('gmNonApple', 'gmNonApple');
load('gmApple3FitMixGauss','gmApple3FitMixGauss');
load('gmNonApple3FitMixGauss','gmNonApple3FitMixGauss')

fileNameTestWithMask = 'testApples/Bbr98ad4z0A-ctgXo3gdwu8-original1';
fileNamesMyTest = {'myTest/testApple1','myTest/testApple2'};
fileNamesTest = {'testApples/Bbr98ad4z0A-ctgXo3gdwu8-original1','testApples/Apples_by_MSR_MikeRyan_flickr1','testApples/audioworm-QKUJj2wmxuI-original1'};

% produces posteriores for test image with mask:
displayPosteriores1(fileNameTestWithMask,gmApple,gmNonApple,1);
% produces posteriores for my first test image:
displayPosteriores1(fileNamesMyTest{1},gmApple,gmNonApple,1);
% produces posteriores for my second test image:
displayPosteriores1(fileNamesMyTest{2},gmApple,gmNonApple,1);
% produces posteriores for test images using 3 gaussians for apple and
% non-apple pixels:
displayPosteriores(fileNamesTest,gmApple3FitMixGauss,gmNonApple3FitMixGauss,0);

% produces ROCs for test image with mask:
produceRocs(fileNameTestWithMask,gmApple,gmNonApple,1);
% produces ROCs for my first test image:
produceRocs(fileNamesMyTest{1},gmApple,gmNonApple,1);
% produces ROCs for my second test image:
produceRocs(fileNamesMyTest{2},gmApple,gmNonApple,1);

function produceRocs(fileName,gmApple,gmNonApple,flag)
[~,nA] = size(gmApple);
[~,nNA] = size(gmNonApple);
figure;
title('Roc Curve');
xlabel('fpr');
ylabel('tpr');
hold on;
legends = cell(nA*nNA,1);
for (i = 1:nA)
    for (j = 1:nNA) 
        curImask = imread(strcat(fileName,'.png'));
        curImask = curImask(:,:,2) > 128;
        [posteriorApple, ~] = calcPosterior(fileName, gmApple{i}, gmNonApple{j},flag);
        auc = produceRoc1(posteriorApple,curImask);
        str = sprintf('#GA:%d - #GNA:%d - AUC:%s',i,j,auc);
        legends{(i-1)*nA+j} = str;
    end
end
legend(legends);
hold off;
end

function displayPosteriores(filesNames,gmApple,gmNonApple,flag)
    %display test image and ground truth;
    %close all;
    [~,num] = size(filesNames);
    figure; set(gcf,'Color',[1 1 1]);
    for (n = 1:num)
        nameFile = filesNames{n};
        % Load image:
        im = double(imread(strcat(nameFile,'.jpg'))) / 255;
        % Load mask:
        % Show image/maks
        subplot(3,2,n*2-1); imagesc(im); axis off; axis image;
        %subplot(n,3,2); imagesc(curImask); colormap(gray); axis off; axis image;
        drawnow;
        % Calculate posteriores:
        [posteriorApple, ~] = calcPosterior(nameFile,gmApple,gmNonApple,flag);
        % Show posteriores:
        clims = [0, 1];
        subplot(3,2,n*2); imagesc(posteriorApple, clims); colormap(gray); axis off; axis image;
    end
end

function displayPosteriores1(fileName,gmsApple,gmsNonApple,flag)
    [~,numgmApple] = size(gmsApple);
    [~,numgmNonApple] = size(gmsNonApple);
    im = double(imread(strcat(fileName,'.jpg'))) / 255;
    figure;
    for (i = 1:numgmApple)
        for (j = 1:numgmNonApple)
            [posteriorApple, ~] = calcPosterior(fileName,gmsApple{i},gmsNonApple{j},flag);
            % Show posteriores:
            clims = [0, 1];
            subplot(numgmApple,numgmApple,j+((i-1)*numgmApple)); imagesc(posteriorApple, clims); colormap(gray); axis off; axis image;
            str = sprintf('GMA:%d-GMNA:%d ',i,j);
            title(str);
            drawnow;
        end
    end
end

function auc = produceRoc1(posteriorApple,Imask)
    thresholds = linspace(0,1);
    [r,c] = size(posteriorApple);
    outputsPostApple = reshape(posteriorApple,1,r*c);
    targets = reshape(Imask,1,r*c);
    positive = sum(targets);
    negative = sum(~targets);
    tpr = zeros(1,100);
    fpr = zeros(1,100);
    for (iter = 1:100)
        above_thres = outputsPostApple >= thresholds(iter);
        true_positive = sum(above_thres.*targets);
        tpr(iter) = true_positive/positive;
        below_thres = outputsPostApple < thresholds(iter);
        true_negative = sum(below_thres.*~targets);
        spc = true_negative/negative;
        fpr(iter) = 1-spc;
    end;
    plot(fpr,tpr,'-');
    drawnow;
    flippedtpr=fliplr(tpr);
    flippedfpr=fliplr(fpr);
    auc = trapz(flippedfpr,flippedtpr);
end


function [posteriorApple, posteriorNonApple] = calcPosterior(nameFile, gmApple, gmNonApple,flag)
    %strcat('testApples/',nameFile,'.jpg')
    
    %let's define priors for whether the pixel is skin or non skin
    priorApple = 0.3;
    priorNonApple = 0.7;

    %now run through the pixels in the image and classify them as being skin or
    %non skin - we will fill in the posterior
    im = double(imread(strcat(nameFile,'.jpg'))) / 255;
    [imY,imX,imZ] = size(im);

    posteriorApple = zeros(imY,imX);
    posteriorNonApple = zeros(imY,imX);
    for (cY = 1:imY); 
        %fprintf('Processing Row %d\n',cY);     
        for (cX = 1:imX);          
            %extract this pixel data
            thisPixelData = squeeze(double(im(cY,cX,:)));
            likeApple = 0;
            likeNonApple = 0;
            %calculate likelihood of this data given apple model
            %if using fgmdist:
            if flag == 1
                for (h = 1:gmApple.NComponents)
                    likeApple = likeApple + gmApple.PComponents(h)*calcGaussianProb(thisPixelData,gmApple.mu(h,:)',gmApple.Sigma(:,:,h));
                end;
                %calculate likelihood of this data given non skin model
                for (h = 1:gmNonApple.NComponents)
                    likeNonApple = likeNonApple + gmNonApple.PComponents(h)*calcGaussianProb(thisPixelData,gmNonApple.mu(h,:)',gmNonApple.Sigma(:,:,h));
                end; 
            else
            %if using fixMixGauss:
                for (h = 1:gmApple.k)
                    likeApple = likeApple + gmNonApple.weight(h)*calcGaussianProb(thisPixelData,gmApple.mean(:,h),gmApple.cov(:,:,h));
                end;
                for (h = 1:gmApple.k)
                    likeNonApple = likeNonApple + gmNonApple.weight(h)*calcGaussianProb(thisPixelData,gmNonApple.mean(:,h),gmNonApple.cov(:,:,h));
                end;
            end;
            %priors using BAYES rule. Replace this: 
            posteriorApple(cY,cX) = (likeApple*priorApple)/(likeApple*priorApple+likeNonApple*priorNonApple);
            posteriorNonApple(cY,cX) = (likeNonApple*priorNonApple)/(likeApple*priorApple+likeNonApple*priorNonApple);
        end;
    end;


end

%the goal of this routine is to evaluate a Gaussian likleihood
function like = calcGaussianProb(data,gaussMean,gaussCov)
[nDim nData] = size(data);
A = 1/((2*pi)^(nData/2)*det(gaussCov)^(0.5));
B = exp(-0.5*transpose(data-gaussMean)*inv(gaussCov)*(data-gaussMean));
like = A*B;
end