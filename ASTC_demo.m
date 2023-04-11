%% *****************************************************************************************************
%%  --------------------- һ������������3D��������Ķ�ģ̬ͼ��ģ��ƥ�䣨ASTC��------------------
%    ---------���Ȳ���һ�־��ȵ���������ȡ������
%    ---------1��������������ǿģ�ͣ���ǿӰ��Ľṹ������
%    ---------2��������������������������ɶ�ά���������ӣ�
%    ---------3���Գ�����ǿ����������������ά������������³�����������ӣ�
%% ------ ���ߣ�Yongxiang  Yao
%%  ************************time:  2022/11��/28��********************************************************
clear;
warning('off');
addpath(genpath('ASTC_Func'));
%%  ����Ӱ��
%%  Optical-infrared ģ̬����
% im_Ref = imread('.\other data\visible-infrared\test6_ref.tif');
% im_Sen = imread('.\other data\visible-infrared\test6_sen.tif');
% CP_Check_file = '.\other data\visible-infrared\VisibletoInfrared_CP.txt';

%%  Optical-LiDAR ģ̬����
im_Ref = imread('.\other data\optical-LiDAR\LiDARintensity1_sen.tif');
im_Sen = imread('.\other data\optical-LiDAR\visible1_ref.tif');
CP_Check_file = '.\other data\optical-LiDAR\LiDARToVisible1.txt';

%%  Optical-Map ģ̬����
% im_Ref = imread('.\other data\optical-Map\optical1_ref.tif');
% im_Sen = imread('.\other data\optical-Map\map1_sen.tif');
% CP_Check_file = '.\other data\optical-Map\opticalToMap1_CP.txt';

%%  Optical-SAR ģ̬����
% im_Ref = imread('.\other data\visible-SAR\visible2_ref.tif');
% im_Sen = imread('.\other data\visible-SAR\SAR2_sen.tif');
% CP_Check_file = '.\other data\visible-SAR\visibleToSAR2_CP.txt';

%%  Night-Optical ģ̬����
% im_Ref = imread('.\other data\visible-Night\pair1-1.jpg');
% im_Sen = imread('.\other data\visible-Night\pair1-2.png');
% CP_Check_file = '.\other data\visible-Night\dayTonight1.txt';

% im_Ref = imread('.\other data\visible-Night\pair2-1.jpg');
% im_Sen = imread('.\other data\visible-Night\pair2-2.jpg');
% CP_Check_file = '.\other data\visible-Night\dayTonight2.txt';

%%  ��ʼ��ʱ
t1=clock;   
%%  ��ʼ����
disthre = 5;                          %   he threshod of match errors the deflaut is 5. for
Detectors = 'BlockFAST';     %   ѡ���������������: 'BlockFAST'�� 'Block_Harris',  'Harris', 'FAST', 'KAZE', 'SURF'
Descriptors = 'ASTC';           %  ��ASTC�� (Ĭ��) 
kpts_nums = 500;                %   ��������Ŀ;
Tw=100;                              %   ģ�崰�ڴ�С;   100
Sw=10;                                  %   �������ڣ�10

%%  template matching using ASTC
[CP_Ref,CP_Sen,CMR,MAPE] = ASTC_match(im_Ref,im_Sen,CP_Check_file,Detectors,Descriptors, disthre,kpts_nums,Tw,Sw);

%%  �ֲ��޳��㷨
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
disp(['RMSE of Matching results: ',num2str(rmse),'  ����']);
%%  
t2=clock;
time=etime(t2,t1);

%%  ͼ���ں�
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