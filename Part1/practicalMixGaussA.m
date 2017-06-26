function r=practicalMixGaussA

%load in test image and ground truth.  Your goal is to segment this image
%to recover the ground truth
im = imread('bob_small.jpeg');
load('bob_GroundTruth_small.mat','gt');

%display test image and ground truth;
close all;
figure; set(gcf,'Color',[1 1 1]);
subplot(1,3,1); imagesc(im); axis off; axis image;
subplot(1,3,2); imagesc(gt); colormap(gray); axis off; axis image;
drawnow;

%load in training data - contains two variables each of size 3 x 10000
%Each column contains RGB values from one pixel in training data
load('RGBSkinNonSkin','RGBSkin','RGBNonSkin');


%fit Gaussian model for skin data
[meanSkin covSkin] = fitGaussianModel(RGBSkin);

%fit Gaussian model for non-skin data
[meanNonSkin covNonSkin] = fitGaussianModel(RGBNonSkin);

%let's define priors for whether the pixel is skin or non skin
priorSkin = 0.3;
priorNonSkin = 0.7;

%now run through the pixels in the image and classify them as being skin or
%non skin - we will fill in the posterior
[imY imX imZ] = size(im);

posteriorSkin = zeros(imY,imX);
for (cY = 1:imY); 
    fprintf('Processing Row %d\n',cY);     
    for (cX = 1:imX);          
        %extract this pixel data
        thisPixelData = squeeze(double(im(cY,cX,:)));
        %calculate likelihood of this data given skin model
        %TO DO - fill in this routine (below)
        likeSkin = calcGaussianProb(thisPixelData,meanSkin,covSkin);
        %calculate likelihood of this data given non skin model
        likeNonSkin = calcGaussianProb(thisPixelData,meanNonSkin,covNonSkin);
        %TO DO (c):  calculate posterior probability from likelihoods and 
        %priors using BAYES rule. Replace this: 
        posteriorSkin(cY,cX) = (likeSkin*priorSkin)/(likeSkin*priorSkin+likeNonSkin*priorNonSkin);
    end;
end;

%draw skin posterior
clims = [0, 1];
subplot(1,3,3); imagesc(posteriorSkin, clims); colormap(gray); axis off; axis image;




%==========================================================================
%==========================================================================

%the goal of this routine is to evaluate a Gaussian likleihood
function like = calcGaussianProb(data,gaussMean,gaussCov)

%TO DO (b) - fill in this routine

[nDim nData] = size(data);
A = 1/((2*pi)^(nData/2)*det(gaussCov)^(0.5));
B = exp(-0.5*transpose(data-transpose(gaussMean))*inv(gaussCov)*(data-transpose(gaussMean)));

like = A*B;



%==========================================================================
%==========================================================================

%the goal of this routine is to return the mean and covariance of a set of
%multidimensaional data.  It is assumed that each column of the 2D array
%data contains a single data point.  The mean vector should be a 3x1 vector
%with the mean RGB value.  The covariance should be a 3x3 covariance
%matrix. See the note at the top, which explains that using mean() is ok,
%but please compute the covariance yourself.
function [meanData covData] = fitGaussianModel(data);

[nDim nData] = size(data);
%TO DO (a): replace this
meanData = sum(transpose(data)) / nData;
A = data-transpose(meanData);
AT = transpose(A);
B = A*AT;
covData = B/(nData-1);

%covData = cov(data)
%calculate mean of data.  You can do this using the MATLAB command 'mean'

%calculate covariance of data.  You should do this yourself to ensure you
%understand how.  Check you have the right answer by comparing with the
%matlab command 'cov'.




