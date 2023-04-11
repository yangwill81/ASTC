function [CP_Ref,CP_Sen,CMR,MAPE] = ASTC_match(Gim_Ref,im_Sen,CP_Check_file,Detectors,Descriptors,disthre,KPT_NUMS,templateSize, searchRad,tranFlag)
% ----------------------------------------------------------------------
% input Parameters: 
%                 im_Ref: the reference image
%                 im_Sen: the sensed imag
%                 H: 单应性矩阵
%                 Detectors: 特征点检测器
%                 Descriptors: 描述子方法
%                 disthre: the threshold of the errors of matching point pair, if the error of the point pair is less than dis, they are regrad as the correct match                 
%                 templateSize: the template size, its minimum value must be more than 20, the default is 100
%                 searchRad: the radius of searchRegion, the default is 10.the maxinum should be lesson than 20
% return vaules:                
%                 CP_Ref: the coordinates ([x,y]) in the reference image
%                 CP_Sen: the coordinates ([x,y]) in the sensed image
%                 CMR: the correct match ratio

if nargin < 3
    disp('the number input parameters must be >= 3 ');
    return;
end
if nargin < 4
    Detectors = 'BlockFAST';        % 选择特征检测器方法: 'GBMS', 'ANMS'， 'B_Harris'
end
if nargin < 5
    Descriptors = 'STCF';        % 选择描述子方法
end
if nargin < 6
    disthre = 1.5;        % the threshod of match errors
end
if nargin < 7
    KPT_NUMS = 500;    % 特征点的数目
end
if nargin < 8
    templateSize =100;% the template size  
end
if nargin < 9
    searchRad = 5;    % the radius of search region
end
if nargin < 10
    tranFlag = 0;    % the radius of search region
end

% tranfer the rgb to gray
[k1,k2,k3] = size(Gim_Ref);
if k3 == 3
    im_Ref = rgb2gray(Gim_Ref);
else
    im_Ref = Gim_Ref;
end
im_Ref = double(im_Ref);

[k1,k2,k3] = size(im_Sen);
if k3 == 3
    im_Sen = rgb2gray(im_Sen);
end
im_Sen = double(im_Sen);
[im_RefH,im_RefW] = size(im_Ref);

templateRad = round(templateSize/2);        %the template radius
marg=templateRad+searchRad+2;              %the boundary. we don't detect tie points out of the boundary
matchRad = templateRad + searchRad;      %the match radius,which is the sum of the template and search radius
C = 0;                                                           %the number of correct match 
CM = 0 ;                                                       %the number of total match 
C_e = 0;                                                         %the number of mismatch
im1 = im_Ref(marg:im_RefH-marg,marg:im_RefW-marg);
%%       选择不同的特征检测器
switch(Detectors)
%%  对比方法
     case 'FAST'   %  多向双层一阶梯度卷积
        a=max(im1(:));  b=min(im1(:));  imn=(im1-b)/(a-b); 
        kpts=detectFASTFeatures(imn,'MinContrast',0.05);
        m1_points=kpts.selectStrongest(KPT_NUMS);
        c =m1_points.Location(:,1);   r =m1_points.Location(:,2);
        points1 =double([r,c] + marg - 1);
     case 'KAZE'   %  多向双层一阶梯度卷积
        a=max(im1(:));  b=min(im1(:));  imn=(im1-b)/(a-b); 
        kpts=detectKAZEFeatures(imn);
        m1_points=kpts.selectStrongest(KPT_NUMS);
        c =round(m1_points.Location(:,1));   r =round(m1_points.Location(:,2));
        points1 =double([r,c] + marg - 1);
     case 'Harris'   %  多向双层一阶梯度卷积
        a=max(im1(:));  b=min(im1(:));  imn=(im1-b)/(a-b); 
        kpts=detectHarrisFeatures(imn,'MinQuality', 0.001);
        m1_points=kpts.selectStrongest(KPT_NUMS);
        c=round(m1_points.Location(:,1));   r =round(m1_points.Location(:,2));
        points1 =double([r,c] + marg - 1);
     case 'SURF'   %  多向双层一阶梯度卷积
        a=max(im1(:));  b=min(im1(:));  imn=(im1-b)/(a-b); 
        kpts=detectSURFFeatures(imn,'MetricThreshold', 0.005);
        m1_points=kpts.selectStrongest(KPT_NUMS);
        c=round(m1_points.Location(:,1));   r =round(m1_points.Location(:,2));
        points1 =double([r,c] + marg - 1);
     case 'BlockFAST'   % 本文方法
        a=max(im1(:));  b=min(im1(:));  imn=(im1-b)/(a-b); 
        m1_points=Block_ExtractFeatures(imn,  40, KPT_NUMS);
        c=round(m1_points(:,1));   r =round(m1_points(:,2));
        points1 =double([r,c] + marg - 1);  
end
pNum = size(points1,1); % the number of interest points
%%  计算多向梯度特征
switch(Descriptors)
    case 'ASTC'                                       %  稠密增强自相似特征
        des_Ref  = Dense_ASTC(im_Ref,8);
        des_Sen = Dense_ASTC(im_Sen,8);
end
% read check points from file;
checkPt = textread(CP_Check_file);
refpt = [checkPt(:,1),checkPt(:,2)]; %the check points in the referencing image
senpt = [checkPt(:,3),checkPt(:,4)]; %the check points in the sensed image

% solve the geometric tranformation parameter
% tran 0:affine, 1: projective, 2: Quadratic polynomial,3: cubic polynomial,the default is 3
tform = [];
if tranFlag == 0
    tform = cp2tform(refpt,senpt,'affine'); 
    T = tform.tdata.T;
elseif tranFlag == 1
    tform = cp2tform(refpt,senpt,'projective');
    T = tform.tdata.T;
    else
    T = solvePoly(refpt,senpt,tranFlag);
end
H = T';%the geometric transformation parameters from im_Ref to im_Sen

for n = 1: pNum
    %the x and y coordinates in the reference image
    X_Ref=points1(n,2);
    Y_Ref=points1(n,1);

    tempCo = [X_Ref,Y_Ref];
    tempCo1 = transferTo(tform,tempCo,H,tranFlag);
    
    %tranformed coordinate (X_Sen_c, Y_Sen_c)
    X_Sen_c = tempCo1(1);
    Y_Sen_c = tempCo1(2);
    X_Sen_c1=round(tempCo1(1));
    Y_Sen_c1 =round(tempCo1(2)); 

    if (X_Sen_c1 < marg+1 | X_Sen_c1 > size(im_Sen,2)-marg | Y_Sen_c1<marg+1 | Y_Sen_c1 > size(im_Sen,1)-marg)
        continue;
    end
    
    DLSS_Ref =single(des_Ref(Y_Ref-matchRad:Y_Ref+matchRad,X_Ref-matchRad:X_Ref+matchRad,:));
    DLSS_Sen =single(des_Sen(Y_Sen_c1-matchRad:Y_Sen_c1+matchRad,X_Sen_c1-matchRad:X_Sen_c1+matchRad,:));
    
    [max_i, max_j] = FFT_3D(DLSS_Ref, DLSS_Sen, matchRad);
    if (~isempty(max_i) || ~isempty(max_j))
    [a,~] = size(max_i);
    if a>1
        continue;
    end
    % the (matchY,matchX) coordinates of match
      Y_match = Y_Sen_c1 + max_i-1;
      X_match = X_Sen_c1 + max_j-1;
    end  
     % calculate the match errors      
       diffY = abs(Y_match-Y_Sen_c);
       diffX = abs(X_match-X_Sen_c);
      diff = sqrt(diffX.^2+diffY.^2);
      MAPE=sum(abs(diffX./X_match)+abs(diffY./Y_match))*100;
      % calculate the numbers of correct match, mismatch and total match
      if diff <= disthre
          C = C+1; % the number of correct matches
          corrp(C,:)=[X_Ref,Y_Ref,X_match,Y_match,MAPE];% the coordinates of correct matches
      else
          C_e = C_e + 1;
          corrp_e(C_e,:) = [X_Ref,X_Ref,X_match,Y_match,MAPE]; % the coordinates of mismatches
      end
      CM = CM + 1;
end
%the correct ratio
CMR = C/CM;
if CMR == 0
    CP_Ref = [];
    CP_Sen = [];
else
    CP_Ref = corrp(:,1:2);
    CP_Sen = corrp(:,3:4);
    MAPE=mean(corrp(:,5));
end

