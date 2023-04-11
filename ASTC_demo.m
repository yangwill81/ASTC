%% *****************************************************************************************************
%%  --------------------- 一种相邻自相似3D卷积特征的多模态图像模板匹配（ASTC）------------------
%    ---------首先采用一种均匀的特征点提取方法；
%    ---------1、引入了特征增强模型，增强影像的结构特征；
%    ---------2、设计了相邻自相似特征，生成多维特征描述子；
%    ---------3、对稠密增强自相似特征进行三维立体卷积，生成鲁棒特征描述子；
%% ------ 作者：Yongxiang  Yao
%%  ************************time:  2022/11月/28日********************************************************
clear;
warning('off');
addpath(genpath('ASTC_Func'));
%%  输入影像
%%  Optical-infrared 模态类型
% im_Ref = imread('.\other data\visible-infrared\test6_ref.tif');
% im_Sen = imread('.\other data\visible-infrared\test6_sen.tif');
% CP_Check_file = '.\other data\visible-infrared\VisibletoInfrared_CP.txt';

%%  Optical-LiDAR 模态类型
im_Ref = imread('.\other data\optical-LiDAR\LiDARintensity1_sen.tif');
im_Sen = imread('.\other data\optical-LiDAR\visible1_ref.tif');
CP_Check_file = '.\other data\optical-LiDAR\LiDARToVisible1.txt';

%%  Optical-Map 模态类型
% im_Ref = imread('.\other data\optical-Map\optical1_ref.tif');
% im_Sen = imread('.\other data\optical-Map\map1_sen.tif');
% CP_Check_file = '.\other data\optical-Map\opticalToMap1_CP.txt';

%%  Optical-SAR 模态类型
% im_Ref = imread('.\other data\visible-SAR\visible2_ref.tif');
% im_Sen = imread('.\other data\visible-SAR\SAR2_sen.tif');
% CP_Check_file = '.\other data\visible-SAR\visibleToSAR2_CP.txt';

%%  Night-Optical 模态类型
% im_Ref = imread('.\other data\visible-Night\pair1-1.jpg');
% im_Sen = imread('.\other data\visible-Night\pair1-2.png');
% CP_Check_file = '.\other data\visible-Night\dayTonight1.txt';

% im_Ref = imread('.\other data\visible-Night\pair2-1.jpg');
% im_Sen = imread('.\other data\visible-Night\pair2-2.jpg');
% CP_Check_file = '.\other data\visible-Night\dayTonight2.txt';

%%  开始计时
t1=clock;   
%%  初始参数
disthre = 5;                          %   he threshod of match errors the deflaut is 5. for
Detectors = 'BlockFAST';     %   选择特征检测器方法: 'BlockFAST'， 'Block_Harris',  'Harris', 'FAST', 'KAZE', 'SURF'
Descriptors = 'ASTC';           %  ‘ASTC’ (默认) 
kpts_nums = 500;                %   特征点数目;
Tw=100;                              %   模板窗口大小;   100
Sw=10;                                  %   搜索窗口：10

%%  template matching using ASTC
[CP_Ref,CP_Sen,CMR,MAPE] = ASTC_match(im_Ref,im_Sen,CP_Check_file,Detectors,Descriptors, disthre,kpts_nums,Tw,Sw);

%%  粗差剔除算法
disp('Outlier removal')   
matchedPoints1 =double(CP_Ref);
matchedPoints2 =double(CP_Sen);
[H,rmse]=FSC(matchedPoints1,matchedPoints2,'affine',3);
Y_=H*[matchedPoints1(:,[1,2])';ones(1,size(matchedPoints1,1))];
Y_(1,:)=Y_(1,:)./Y_(3,:);
Y_(2,:)=Y_(2,:)./Y_(3,:);
E=sqrt(sum((Y_(1:2,:)-matchedPoints2(:,[1,2])').^2));
inliersIndex=E < 3;
clearedPoints1 = matchedPoints1(inliersIndex, :);
clearedPoints2 = matchedPoints2(inliersIndex, :);
uni1=[clearedPoints1(:,[1,2]),clearedPoints2(:,[1,2])];
[~,i,~]=unique(uni1,'rows','first');
inliersPoints1=clearedPoints1(sort(i)',:);
inliersPoints2=clearedPoints2(sort(i)',:);
disp(['RMSE of Matching results: ',num2str(rmse),'  像素']);
%%  
t2=clock;
time=etime(t2,t1);

%%  图像融合
image_fusion(im_Ref, im_Sen,inv(H));

corrRefPt = inliersPoints1;
corrSenPt = inliersPoints2;
RCM=(size(corrRefPt,1)/kpts_nums)*100;
disp(['The RCM is :',num2str(RCM),' %']);   
disp(['The numbers in ASTC algorithm matching points :']);
disp(size(corrRefPt,1));
disp(['The total time spent in ASTC algorithm matching :',num2str(time),' S']);    

%%  
figure;
imshow(im_Ref),hold on;
plot(corrRefPt(:,1),corrRefPt(:,2),'go','MarkerEdgeColor','k','MarkerFaceColor','y','MarkerSize',5.0);hold on;
title('reference image');
figure;
imshow(im_Sen),hold on;
plot(corrSenPt(:,1),corrSenPt(:,2),'go','MarkerEdgeColor','k','MarkerFaceColor','y','MarkerSize',5.0);hold on;
title('sensed image');