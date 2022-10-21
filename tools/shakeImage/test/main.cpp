#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include<string>
#include <work.h>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\calib3d\calib3d.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/opencv_modules.hpp>

#include"sift.h"
#include"display.h"
#include"match.h"
#include "ORBExtractor.h"
#include<stdlib.h>
#include<direct.h>
using namespace cv;
//#pragma comment(lib, "ShakeImage.lib")
typedef unsigned char*(*PLoadFrame)(unsigned char* currentFrame, unsigned char* preFrame, int rows, int cols, int channels);
typedef bool(*PTestVideoShake)(char* intputfilepath, char* outputfilepath);
int image_idx = 0;
Mat calcHMat1(Mat image_1, Mat image_2, int nFeatures, float scaleFactor, int nLevels)
{
	string change_model = string("perspective");
	//Mat image_1, image_2;

	//创建文件夹保存图像
	char* newfile = ".\\image_save";
	_mkdir(newfile);

	//参考图像特征点检测和描述
	vector<vector<Mat>> gauss_pyr_1, dog_pyr_1;
	vector<KeyPoint> keypoints_1;
	Mat descriptors_1;

	//待配准图像特征点检测和描述
	vector<vector<Mat>> gauss_pyr_2, dog_pyr_2;
	vector<KeyPoint> keypoints_2;
	Mat descriptors_2;
	Ptr<DescriptorMatcher> matcher;


	//算法运行总时间开始计时
	double total_count_beg = (double)getTickCount();


	ORB_SLAM2::ORBextractor orb(nFeatures, scaleFactor, nLevels, 60, 60);
	orb(image_1, Mat(), keypoints_1, descriptors_1, 0);

	double before_orb2 = (double)getTickCount();

	ORB_SLAM2::ORBextractor orb2(nFeatures, scaleFactor, nLevels, 60, 60);
	orb2(image_2, Mat(), keypoints_2, descriptors_2, 0);


	//最近邻与次近邻距离比匹配
	double before_knn_match = (double)getTickCount();
	std::vector<vector<DMatch>> dmatches;

	Mat pos1, pos2;
	pos1.create((int)keypoints_1.size(), 2, CV_32F);
	pos2.create((int)keypoints_2.size(), 2, CV_32F);

	for (int i = 0; i < keypoints_1.size(); i++)
	{
		pos1.at<float>(i, 0) = keypoints_1.at(i).pt.x;
		pos1.at<float>(i, 1) = keypoints_1.at(i).pt.y;
	}

	for (int i = 0; i < keypoints_2.size(); i++)
	{
		pos2.at<float>(i, 0) = keypoints_2.at(i).pt.x;
		pos2.at<float>(i, 1) = keypoints_2.at(i).pt.y;
	}

	matcher = new BFMatcher(NORM_L2);
	matcher->knnMatch(pos1, pos2, dmatches, 10);

#ifdef _DEBUG
	//cout << "根据距离匹配的的关键点对： " << dmatches.size() << "\n";
#endif 


	double after_knn_match = (double)getTickCount();
	//match_des(descriptors_1, descriptors_2, dmatches, COS);
	//filterKPbyMatch(keypoints_1, dmatches);

#ifdef _DEBUG
	//cout << "删除mask后的匹配数： " << dmatches.size() << "\n";
#endif 

	//vector<Point2f> seqKeyPoints1, seqKeyPoints2;
	//for (int i = 0; i < right_matches.size(); i++)
	//{
	//	seqKeyPoints1.push_back(keypoints_1.at(right_matches.at(i).queryIdx).pt);
	//	seqKeyPoints2.push_back(keypoints_2.at(right_matches.at(i).trainIdx).pt);
	//}
	//Mat homography = findHomography(seqKeyPoints1, seqKeyPoints2, RANSAC);

	//求解Homography矩阵
	Mat matched_lines;
	vector<DMatch> right_matches;
	double after_h_match = (double)getTickCount();
	Mat homography = match(dmatches, keypoints_1, keypoints_2, change_model,
		right_matches, after_h_match);

#ifdef _DEBUG

	//绘制正确匹配点对连线图
	vector<char> rightMatchMask;
	for (int i = 0; i < right_matches.size(); i++)
	{
		rightMatchMask.push_back(0);
	}
	drawMatches(image_1, keypoints_1, image_2, keypoints_2, right_matches, matched_lines, Scalar(0, 255, 0), Scalar(0, 0, 255),
		vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::imwrite(".\\image_save\\正确匹配点对" + to_string(image_idx) + ".jpg", matched_lines);

	////校正
	//Mat warpImage2;
	//cv::warpPerspective(image2, warpImage2, homography, image_2.size());
	//cv::imwrite(".\\image_save\\校正后的图像2.jpg", warpImage2);

	////图像融合
	//homography.convertTo(homography, CV_32F);

	//Mat fusion_image, mosaic_image, regist_image;
	//image_fusion(image_1, warpImage2, homography, fusion_image, mosaic_image, regist_image);
	//String image_name = ".\\image_save\\融合后的镶嵌图像.jpg";
	//cv::imwrite(image_name, mosaic_image);

#endif // _DEBUG

	return homography;
}
int main(int argc, char *argv[]) {
	//HMODULE hinst = LoadLibraryA(("F:\\work\\workSpace\\minanqiang\\imageregistration_new\\image_registration\\build\\Debug\\Dll1.dll"));
	std::string xx = "F:\\work\\workSpace\\minanqiang\\imageregistration_new\\image_registration\\build\\x64\\Debug\\Dll1.dll";
	HMODULE hinst1 = LoadLibraryA(xx.c_str());
	 //xx = "F:\\SimOneAPI.dll";
	
	 //hinst1 = LoadLibraryA(xx.c_str());
	 //xx = "F:\\Dll1.dll";
	 //hinst1 = LoadLibraryA(xx.c_str());
	xx = "F:\\work\\workSpace\\minanqiang\\imageregistration_new\\image_registration\\build\\Debug\\ShakeImage.dll";
	 hinst1 = LoadLibraryA(xx.c_str());

	 PLoadFrame pPLoadFrame;
	pPLoadFrame = (PLoadFrame)GetProcAddress(hinst1, "LoadFrame");
	 VideoCapture inputVideo("F:\\2022-09-24 16-24-54.mkv");
	 if (!inputVideo.isOpened())
	 {
		 //std::cout << "Error opening video stream" << "\n";
		 return false;
	 }
	 int currentFrame = -1;
	 Mat pre, current;
	 Mat gray_first, gray_current;
	 float unit_mat[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
	 Mat lastHomography(3, 3, CV_32F, unit_mat);
	 //Mat homography;

	 //int ex = static_cast<int>(inputVideo.get(CAP_PROP_FOURCC));     // Get Codec Type- Int form
	 double fps = inputVideo.get(CAP_PROP_FPS);
	 Size size = Size(inputVideo.get(CAP_PROP_FRAME_WIDTH), inputVideo.get(CAP_PROP_FRAME_HEIGHT));
	 int total_frames = inputVideo.get(CAP_PROP_FRAME_COUNT);

	 int ex = VideoWriter::fourcc('X', 'V', 'I', 'D');
	 cv::VideoWriter outputVideo("F:\\image\\output.avi", ex, fps, size);
	 while (1)
	 {
		 Mat frame;
		 int64 iter_start = getTickCount();

		 inputVideo >> frame;
		 if (frame.empty())
		 {
			 break;
		 }
		
		 //Mat image_1 = frame.clone();
		 //Mat image_2 = frame.clone();
		 //Mat gray_image1, gray_image2;
		 //cvtColor(image_1, gray_image1, cv::COLOR_BGR2GRAY);
		 //cvtColor(image_2, gray_image2, cv::COLOR_BGR2GRAY);
		 
		 currentFrame++;
		 printf("currentFrame:%d\n", currentFrame);
		 if (currentFrame == 0)
		 {
			 pre = frame.clone();
			 cvtColor(pre, gray_first, cv::COLOR_BGR2GRAY);
			 cv::imwrite("F:\\2.jpg", gray_first);
			 continue;
		 }
		 else
		 {
			 Mat forecast;
			 current = frame.clone();

			 

			 warpPerspective(current, forecast, lastHomography, size);
			 cvtColor(forecast, gray_current, cv::COLOR_BGR2GRAY);
			 printf("current frame:%d,total_frame:%d\n", currentFrame + 1, total_frames);
			 unsigned char* data = pPLoadFrame(gray_first.data, gray_current.data, gray_current.size().height, gray_current.size().width, 3);
			 
			 Mat homography = Mat(Size(3, 3), CV_32FC1,Scalar(0));
			 homography.data = data;

			 //cout << "M2= " << endl << " " << homography << endl << endl;

			 Mat result;
			 warpPerspective(forecast, result, homography, size);
			 
			 //std::string file_name = "F:\\image\\result_"+ std::to_string(currentFrame)+".jpg";
			 //cv::imwrite(file_name, result);

			//std::string file_name = "F:\\image\\result_gray_first"+ std::to_string(currentFrame)+".jpg";
			// cv::imwrite(file_name, gray_first);

			// file_name = "F:\\image\\result_gray_current" + std::to_string(currentFrame) + ".jpg";
			// cv::imwrite(file_name, gray_current);

			 outputVideo.write(result);
	
			 Mat tmp;
			 cout << "before lastHomography= " << endl << " " << lastHomography << endl << endl;
			 cout << "before homography= " << endl << " " << homography << endl << endl;
			 tmp = homography * lastHomography;
			 lastHomography = tmp;
			 cout << "after lastHomography= " << endl << " " << lastHomography << endl << endl;
			
		 }
		 
		 
		 //Mat tmp1 = cv::imread("F:\\1.jpg");

		 //Mat current = Mat(Size(rows, cols), CV_8UC1, Scalar(255));
		 //current.data = preFrame;
		 //cv::imwrite("F:\\2.jpg", current);

		 //Mat tmp2 = cv::imread("F:\\2.jpg");


		 //float unit_mat[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
		 //Mat lastHomography(3, 3, CV_32F, unit_mat);
		 //Mat forecast;
		 //image_2 = frame.clone();
		 //warpPerspective(image_2, forecast, lastHomography, size);
		 //cvtColor(forecast, gray_image2, cv::COLOR_BGR2GRAY);

		 //Mat homography = calcHMat1(tmp1, tmp2, 1200, 1.2, 1);
		 //for (int i = 0; i < homography.rows; i++) {
			// for (int j = 0; j < homography.cols; j++) {
			//	 float bgr = homography.at<float>(i, j);
			//	 printf("bgr");
			//	 //printf("b = %d, g = %d, r = %d\n", bgr[0], bgr[1], bgr[2]);
			// }
		 //}
		 //homography.data;
		 //cout << "M= " << endl << " " << homography << endl << endl;
		 //////校正
		 //Mat warpImage2;
		 //warpPerspective(tmp1, warpImage2, homography, size);

		 //cv::imwrite("F:\\3.jpg", warpImage2);
		 ////outputVideo.write(warpImage2);
		 //int64 afterWriteFrame = getTickCount();

		 //int64 calc_finish = getTickCount();

		 
	 }
	 inputVideo.release();
	 outputVideo.release();
	 /*PTestVideoShake pTestVideoShake;
	 pTestVideoShake = (PTestVideoShake)GetProcAddress(hinst1, "TestVideoShake");
	 bool x = pTestVideoShake("F:\\work\\workSpace\\minanqiang\\imageregistration_new\\007.mp4",
		 "F:\\work\\workSpace\\minanqiang\\imageregistration_new\\007.01.mp4");*/
	 //PLoadFrame pPLoadFrame;
	 //pPLoadFrame = (PLoadFrame)GetProcAddress(hinst1, "LoadFrame");
	 //bool x = pPLoadFrame(0);
	//if (NULL == hinst1)
	//{
	//	//资源加载失败!
	//	return TRUE;
	//}
	return true;
}