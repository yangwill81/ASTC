function Im_Kpts =Block_ExtractFeatures(Image, blockSize, KPT_NUMS)

%%   分块提取特征点
if nargin < 2
    disp('the number input parameters must be >= 2 ');
    return;
end
if nargin < 2
    blockSize = 40;        % % 设置分块的大小'
end
if nargin < 3
    KPT_NUMS = 500;        % 选择特征检测器方法: 'GBMS', 'ANMS'， 'B_Harris'
end
%%  相位特征计算
[Im,~,~]=PC_phasecong(Image);
% figure, imshow(Im,[]);

%%   分块提取策略
[rows, cols] = size(Im);
% 计算分块的行数和列数
numBlockRows = ceil(rows / blockSize);
numBlockCols = ceil(cols / blockSize);

Block_KPT_NUMS=round(KPT_NUMS/(numBlockRows*numBlockCols))+1;

% 初始化特征点的位置和描述符
Points = [];
% 循环处理每个分块
for i = 1:numBlockRows
    for j = 1:numBlockCols
        % 计算分块的起始行和结束行
        startRow = (i-1) * blockSize + 1;
        endRow = min(i * blockSize, rows);
        % 计算分块的起始列和结束列
        startCol = (j-1) * blockSize + 1;
        endCol = min(j * blockSize, cols);
        % 提取分块
        block = Im(startRow:endRow, startCol:endCol);
        pointsBlock = detectFASTFeatures(block,'MinContrast', 0.0005);
        B_points=pointsBlock.selectStrongest(Block_KPT_NUMS);
        % 更新特征点的位置
        Points = [Points; B_points.Location + [startCol-1 startRow-1],B_points.Metric];
    end
end
%% 提取所需要的特征点数量
Im_Kpts = sortrows(Points,3,'descend');
Im_Kpts =Im_Kpts(1:KPT_NUMS,[1,2]);
end

%%  相位一致性梯度主方向计算
function[MM,phaseCongruency,or]=PC_phasecong(im, nscale, norient)

if nargin < 2
    nscale          = 5;     % Number of wavelet scales.
end
if nargin < 3
    norient         = 6;     % Number of filter orientations.
end
if nargin < 4
    noiseMode = 1;
end
if nargin < 5
    minWaveLength   = 3;     % Wavelength of smallest scale filter.
end
if nargin < 6
    mult            = 2.0;     % Scaling factor between successive filters.

end
if nargin < 7
    sigmaOnf        = 0.55;  % Ratio of the standard deviation of the
                             % Gaussian describing the log Gabor filter's transfer function 
			     % in the frequency domain to the filter center frequency.
end
if nargin < 8
    dThetaOnSigma   = 1.7;   % Ratio of angular interval between filter orientations
			     % and the standard deviation of the angular Gaussian
			     % function used to construct filters in the
                             % freq. plane.
end
if nargin < 9
    k               = 3.0;   % No of standard deviations of the noise energy beyond the
			     % mean at which we set the noise threshold point.
			     % standard deviation to its maximum effect
                             % on Energy.
end
if nargin < 10
    cutOff          = 0.4;   % The fractional measure of frequency spread
                             % below which phase congruency values get penalized.
end
   
g               = 10;    % Controls the sharpness of the transition in the sigmoid
                         % function used to weight phase congruency for frequency
                         % spread.
epsilon         = .0001; % Used to prevent division by zero.
thetaSigma = pi/norient/dThetaOnSigma;  % Calculate the standard deviation of the
imagefft = single(fft2(im));                    % Fourier transform of image
sze = size(imagefft);
rows = sze(1);
cols = sze(2);
zero = single(zeros(sze));
totalEnergy = zero;                     % Matrix for accumulating weighted phase 
totalSumAn  = zero;                     % Matrix for accumulating filter response
estMeanE2n = [];

covx2 = zero;                     % Matrices for covariance data
covy2 = zero;
covxy = zero;
EnergyV2 = zero;
EnergyV3 = zero;

% Pre-compute some stuff to speed up filter construction

x = single(ones(rows,1) * (-cols/2 : (cols/2 - 1))/(cols/2));  
y = single((-rows/2 : (rows/2 - 1))' * ones(1,cols)/(rows/2));
radius = sqrt(x.^2 + y.^2);       % Matrix values contain *normalised* radius from centre.
radius(round(rows/2+1),round(cols/2+1)) = 1; % Get rid of the 0 radius value in the middle 
theta = single(atan2(-y,x));              % Matrix values contain polar angle.
sintheta = sin(theta);
costheta = cos(theta);
clear x; clear y; clear theta; clear im;     % save a little memory

% The main loop...
for o = 1:norient,                   % For each orientation.
  angl = (o-1)*pi/norient;           % Calculate filter angle.
  wavelength = minWaveLength;        % Initialize filter wavelength.
  sumE_ThisOrient   = zero;          % Initialize accumulator matrices.
  sumO_ThisOrient   = zero;  
  sumO_ThisOrient1   = zero;        % the odd energy expect the smallest wave scale
  sumAn_ThisOrient  = zero;      
  Energy_ThisOrient = zero;      
  EOArray = single([]);          % Array of complex convolution images - one for each scale.
  ifftFilterArray = single([]);  % Array of inverse FFTs of filters

  ds = sintheta * cos(angl) - costheta * sin(angl); % Difference in sine.
  dc = costheta * cos(angl) + sintheta * sin(angl); % Difference in cosine.
  dtheta = abs(atan2(ds,dc));                           % Absolute angular distance.
  spread = exp((-dtheta.^2) / (2 * thetaSigma^2));      % Calculate the angular filter component.

  clear ds;clear dc;clear dtheta;
  for s = 1:nscale,                  % For each scale.
    fo = 1.0/wavelength;                  % Centre frequency of filter.
    rfo = fo/0.5;                         % Normalised radius from centre of frequency plane 
    logGabor = exp((-(log(radius/rfo)).^2) / (2 * log(sigmaOnf)^2));  
    logGabor(round(rows/2+1),round(cols/2+1)) = 0; % Set the value at the center of the filter
    filter = logGabor .* spread;          % Multiply by the angular spread to get the filter.
    filter = fftshift(filter);            % Swap quadrants to move zero frequency 
    clear logGabor;
    ifftFilt = real(ifft2(filter))*sqrt(rows*cols);  % Note rescaling to match power
    ifftFilterArray = single([ifftFilterArray ifftFilt]);    % record ifft2 of filter
    clear ifftFilt;
    EOfft = imagefft .* filter;           % Do the convolution.
    EO = single(ifft2(EOfft));                    % Back transform.
    clear EOfft;

    EOArray = single([EOArray, EO]);              % Record convolution result
    An = abs(EO);                         % Amplitude of even & odd filter response.

    sumAn_ThisOrient = single(sumAn_ThisOrient + An);     % Sum of amplitude responses.
    sumE_ThisOrient = single(sumE_ThisOrient + real(EO)); % Sum of even filter convolution results.
    sumO_ThisOrient = single(sumO_ThisOrient + imag(EO)); % Sum of odd filter convolution results.
    if s>1;
        sumO_ThisOrient1 = single(sumO_ThisOrient1 + imag(EO));%sum the odd amplitude response except the smallest scale
    end
    if s == 1
       maxSumO = sumO_ThisOrient; %Record the maximum odd amplitude responses
    else
        maxSumO = max(maxSumO,sumO_ThisOrient);
    end
    if s == 1                             % Record the maximum An over all scales
      maxAn = An;
    else
      maxAn = max(maxAn, An);
    end
    
    if s==1
      EM_n = sum(sum(filter.^2));           % Record mean squared filter value at smallest
    end                                     % scale. This is used for noise estimation.

    wavelength = wavelength * mult;% Finally calculate Wavelength of next filter
    
    clear An; clear filter;
  end                                       % ... and process the next scale

  XEnergy = sqrt(sumE_ThisOrient.^2 + sumO_ThisOrient.^2) + epsilon;   
  MeanE = sumE_ThisOrient ./ XEnergy; 
  MeanO = sumO_ThisOrient ./ XEnergy; 
  clear XEnergy;
  for s = 1:nscale,       
      EO = submat(EOArray,s,cols);  % Extract even and odd filter 
      E = real(EO); O = imag(EO);
      Energy_ThisOrient = Energy_ThisOrient ...
        + E.*MeanE + O.*MeanO - abs(E.*MeanO - O.*MeanE);
  end
  clear EO;clear E; clear O;clear MeanE; clear MeanO;

  medianE2n = median(reshape(abs(submat(EOArray,1,cols)).^2,1,rows*cols));
  meanE2n = -medianE2n/log(0.5);
  estMeanE2n = [estMeanE2n meanE2n];
  noisePower = meanE2n/EM_n;                       % Estimate of noise power.
  clear meanE2n;clear medianE2n; clear meanE2n;

  EstSumAn2 = zero;
  for s = 1:nscale
    EstSumAn2 = EstSumAn2+submat(ifftFilterArray,s,cols).^2;
  end

  EstSumAiAj = zero;
  for si = 1:(nscale-1)
    for sj = (si+1):nscale
      EstSumAiAj = EstSumAiAj + submat(ifftFilterArray,si,cols).*submat(ifftFilterArray,sj,cols);
    end
  end

  EstNoiseEnergy2 = 2*noisePower*sum(sum(EstSumAn2)) + 4*noisePower*sum(sum(EstSumAiAj));
  
  clear EstSumAn2;
  tau = sqrt(EstNoiseEnergy2/2);                     % Rayleigh parameter
  EstNoiseEnergy = tau*sqrt(pi/2);                   % Expected value of noise energy
  EstNoiseEnergySigma = sqrt( (2-pi/2)*tau^2 );

  T =  EstNoiseEnergy + k*EstNoiseEnergySigma;       % Noise threshold
  
  clear EstNoiseEnergy; clear EstNoiseEnergySigma; clear tau;
  clear EstNoiseEnergy2;clear EstSumAiAj;clear noisePower; 

  T = T/1.7;        % Empirical rescaling of the estimated noise effect to 

  Energy_ThisOrient = max(Energy_ThisOrient - T, zero);  % Apply noise threshold
  width = sumAn_ThisOrient ./ (maxAn + epsilon) / nscale;    
  % Now calculate the sigmoidal weighting function for this orientation.
  weight = 1.0 ./ (1 + exp( (cutOff - width)*g)); 
  Energy_ThisOrient =   weight.*Energy_ThisOrient;
  clear weight;clear width;
  totalSumAn  = totalSumAn + sumAn_ThisOrient;%分母
  totalEnergy = totalEnergy + Energy_ThisOrient;%分子
  
  PC{o} = Energy_ThisOrient./sumAn_ThisOrient;
  
  % Build up covariance data for every point
  covx = PC{o}*cos(angl);
  covy = PC{o}*sin(angl);
  covx2 = covx2 + covx.^2;
  covy2 = covy2 + covy.^2;
  covxy = covxy + covx.*covy;

EnergyV2 = EnergyV2 + cos(angl)*sumO_ThisOrient;
EnergyV3 = EnergyV3 + sin(angl)*sumO_ThisOrient;
  
  clear sumAn_ThisOrient; clear Energy_ThisOrient; clear sumO_ThisOrient;
  clear sumO_ThisOrient; clear spread; clear EOArray; clear ifftFilterArray;

end  % For each orientation

    % First normalise covariance values by the number of orientations/2
    covx2 = covx2/(norient/2);
    covy2 = covy2/(norient/2);
    covxy = 4*covxy/norient;   % This gives us 2*covxy/(norient/2)
    denom = sqrt(covxy.^2 + (covx2-covy2).^2)+epsilon;
    M = (covy2+covx2 + denom)/2;          % Maximum moment
    m = (covy2+covx2 - denom)/2;          % Minimum moment

    MM=double(M+m)/2;
%     a=max(MM(:));  b=min(MM(:));  MM=(MM-b)/(a-b); 
    
    phaseCongruency = double(totalEnergy ./ (totalSumAn + epsilon));%used the atan (EnergyV3,-EnergyV2) to abain phase cogruency magnitude 
    or = double(atan(EnergyV3./(-EnergyV2)));%note the y direction is reverse
    
end    
    
    
function a = submat(big,i,cols)

a = big(:,((i-1)*cols+1):(i*cols));
end
