%close all open plots
close all;

[appleTrain,nonAppleTrain] = getApplesData();
save('RGBAppleTrain.mat','appleTrain');
save('RGBNonAppleTrain.mat','nonAppleTrain');


function [appleTrain, nonAppleTrain] = buildTrainingData(data,maskI)
    [ydim,xdim,zdim] = size(data);
    % Adding 0.1 to each rgb value of each pixel so that white pixels
    % belonging to apples wont be lost
    dataTemp = data+0.1;
    applePixelsVec = transpose(reshape(dataTemp,ydim*xdim,zdim));
    maskVec = transpose(reshape(maskI,ydim*xdim,1));
    applePixelsArray = applePixelsVec.*maskVec;
    appleTrain = [];
    for (pixel = 1:ydim*xdim)
        if (applePixelsArray(1,pixel) ~= 0 && applePixelsArray(2,pixel) ~= 0 && applePixelsArray(3,pixel) ~= 0)
            newElem = [applePixelsArray(1,pixel) applePixelsArray(2,pixel) applePixelsArray(3,pixel)];
            appleTrain(:,end+1) = transpose(newElem);
        end
    end
    appleTrain = appleTrain - 0.1;
    [~,numApplePixels] = size(appleTrain);
    
    nonApplePixelsVec = transpose(reshape(dataTemp,ydim*xdim,zdim));
    invMask = ~maskI;
    invMaskVec = transpose(reshape(invMask,ydim*xdim,1));
    nonApplePixelsArray = nonApplePixelsVec.*invMaskVec;
    nonAppleTrain = [];
    for (pixel = 1:ydim*xdim)
        if (nonApplePixelsArray(1,pixel) ~= 0 && nonApplePixelsArray(2,pixel) ~= 0 && nonApplePixelsArray(3,pixel) ~= 0)
            newElem = [nonApplePixelsArray(1,pixel) nonApplePixelsArray(2,pixel) nonApplePixelsArray(3,pixel)];
            nonAppleTrain(:,end+1) = transpose(newElem);
        end
    end
    nonAppleTrain = nonAppleTrain - 0.1;
    [~,numNonApplePixels] = size(nonAppleTrain);
    
    if (numApplePixels + numNonApplePixels ~= xdim*ydim)
        disp('Num of apple and non apple pixels dont correspond to total number of pixels!')
    end
end


function [appleTrain,nonAppleTrain] = getApplesData()

    if( ~exist('apples', 'dir') || ~exist('testApples', 'dir') )
        display('Please change current directory to the parent folder of both apples/ and testApples/');
    end

    % Note that cells are accessed using curly-brackets {} instead of parentheses ().
    Iapples = cell(3,1);
    Iapples{1} = 'apples/Apples_by_kightp_Pat_Knight_flickr.jpg';
    Iapples{2} = 'apples/ApplesAndPears_by_srqpix_ClydeRobinson.jpg';
    Iapples{3} = 'apples/bobbing-for-apples.jpg';

    IapplesMasks = cell(3,1);
    IapplesMasks{1} = 'apples/Apples_by_kightp_Pat_Knight_flickr.png';
    IapplesMasks{2} = 'apples/ApplesAndPears_by_srqpix_ClydeRobinson.png';
    IapplesMasks{3} = 'apples/bobbing-for-apples.png';
    appleTrain = [];
    nonAppleTrain = [];
    for (iImage = 1:3)
        curI = double(imread(  Iapples{iImage}   )) / 255;
        curImask = imread(  IapplesMasks{iImage}   );
        curImask = curImask(:,:,2) > 128;  % Picked green-channel arbitrarily.
        [appleTrainNew, nonAppleTrainNew] = buildTrainingData(curI,curImask);
        appleTrain = horzcat(appleTrain,appleTrainNew);
        nonAppleTrain = horzcat(nonAppleTrain,nonAppleTrainNew);
    end;
    size(appleTrain)
    size(nonAppleTrain)
end


function imHSL = rgbtohsl(fileName)
    imRGB = double(imread(strcat(fileName,'.jpg'))) / 255;
    [imY,imX,imZ] = size(imRGB);
    imHSL = zeros(imY,imX,imZ);
    for (y = 1:imY)
        for (x = 1:imX)
            pixelRGB = imRGB(imY,imX,:);
            minVal = min(pixelRGB);
            [maxVal,i] = max(pixelRGB);
            lum = (minVal+maxVal)/2;
            if minVal == maxVal
                sat = 0;
            elseif lum < 0.5
                sat = (maxVal-minVal)/(maxVal+minVal);
            else
                sat = (maxVal-minVal)/(2.0-maxVal-minVal);
            end
            if i == 1
                hue = (pixelRGB(2)-pixelRGB(3))/(maxVal-minVal);
            elseif i == 2
                hue = 2.0+(pixelRGB(3)-pixelRGB(1))/(maxVal-minVal); 
            else
                hue = 4.0+(pixelRGB(1)-pixelRGB(2))/(maxVal-minVal); 
            end
            imHSL(y,x,:) = [lum,sat,hue];
        end
    end
end

