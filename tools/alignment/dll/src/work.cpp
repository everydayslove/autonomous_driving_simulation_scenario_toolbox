#include<stdio.h>
#include <thread>
#include"work.h"
#include <opencv2/opencv.hpp>
#include"sift.h"
#include"display.h"
#include"match.h"
#include "ORBExtractor.h"

#include<opencv2\highgui\highgui.hpp>
#include<opencv2\calib3d\calib3d.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/opencv_modules.hpp>
#include <iostream>
#include<fstream>
#include<stdlib.h>
#include<direct.h>
#include<iomanip>
#include<filesystem>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
using namespace cv;
using namespace std;
// MatSO声明
class MatSO
{
public:
	// 该函数接收uchar数据，转为Mat数据，显示，然后再返回uchar数据
	uchar *get_mat_and_return_uchar(uchar *matrix, int rows, int cols, int channels);

	// 该函数接收uchar数据，转为Mat数据，显示，然后再返回buffle数据
	uchar *get_mat_and_return_buffer(uchar *matrix, int rows, int cols, int channels);
};


// mat数据接收并以uchar返回
uchar *MatSO::get_mat_and_return_uchar(uchar *matrix, int rows, int cols, int channels)
{
	// 将接收的uchar转为Mat
	cout << "rows=" << rows << "\ncols=" << cols << "\nchannels=" << channels;
	//Mat input_mat = Mat(Size(cols, rows), CV_8UC3, Scalar(255, 255, 255));
	Mat input_mat = Mat(Size(cols, rows), CV_8UC1, Scalar(0));
	input_mat.data = matrix;

	// 显示Mat
	//imshow("input_mat", input_mat);
	//cv::waitKey(0);

	// 将Mat转为uchar类型并返回
	// 注意：要记住这里Mat的rol、row和channels，在python中才能正确接收
	uchar *s = input_mat.data; // Mat转ucahr*
	return s;
}


// mat数据接收并以buffer返回
uchar *MatSO::get_mat_and_return_buffer(uchar *matrix, int rows, int cols, int channels)
{
	// 将接收的uchar转为Mat
	cout << "rows=" << rows << "\ncols=" << cols << "\nchannels=" << channels;
	Mat input_mat = Mat(Size(cols, rows), CV_8UC3, Scalar(255, 255, 255));
	input_mat.data = matrix;

	// 显示Mat
	//imshow("input_mat", input_mat);
	//cv::waitKey(0);

	// 将Mat转为buffer并返回
	// 注意：要记住这里Mat的rol、row和channels，在python中才能正确接收
	int height = input_mat.cols;
	int width = input_mat.rows;
	uchar *buffer = (uchar *)malloc(sizeof(uchar) * height * width * 3);
	memcpy(buffer, input_mat.data, height * width * 3);
	return buffer;
}
//int image_idx = 0;
Mat calcHMat1(Mat image_1, Mat image_2, int nFeatures, float scaleFactor, int nLevels)
{
	string change_model = string("perspective");
	//Mat image_1, image_2;

	//创建文件夹保存图像
	//char* newfile = ".\\image_save";
	//_mkdir(newfile);

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
	//vector<char> rightMatchMask;
	//for (int i = 0; i < right_matches.size(); i++)
	//{
	//	rightMatchMask.push_back(0);
	//}
	//drawMatches(image_1, keypoints_1, image_2, keypoints_2, right_matches, matched_lines, Scalar(0, 255, 0), Scalar(0, 0, 255),
	//	vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//cv::imwrite(".\\image_save\\正确匹配点对" + to_string(image_idx) + ".jpg", matched_lines);

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
int gcurrentFrame = -1;
float unit_mat[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
Mat lastHomography(3, 3, CV_32F, unit_mat);

bool test(Mat& currentFrame,Mat& homography) {

	shared_ptr<Mat> homography_ptr;

	
	//Mat homography;

	//handle the first frame
	vector<KeyPoint> keypoints_1;
	Mat gray_image1, gray_image2;

	int64 begin = getTickCount();
	int64 lastFrame = getTickCount();

	Size size = Size(currentFrame.size().width, currentFrame.size().height);
	//while (1)
	{
		Mat image_1, image_2;
		int64 iter_start = getTickCount();


		gcurrentFrame++;

		if (currentFrame.size().height == 0)
		{
			std::cout << "Read video failed." << "\n";
			return false;
		}

		if (gcurrentFrame == 0)
		{
			image_1 = currentFrame.clone();
			std::cout << "\n";
			cvtColor(image_1, gray_image1, cv::COLOR_BGR2GRAY);
		}
		else
		{
			Mat forecast;
			image_2 = currentFrame.clone();
			warpPerspective(image_2, forecast, lastHomography, size);
			cvtColor(forecast, gray_image2, cv::COLOR_BGR2GRAY);

			int64 calc_start = getTickCount();
			homography = calcHMat1(gray_image1, gray_image2, 1200, 1.2, 1);
			int64 calc_finish = getTickCount();

			Mat tmp;
			tmp = homography * lastHomography;
			lastHomography = tmp;
		}
	}

	//outputVideo.release();
	std::cout << "\n" << "Total time cost :" << (double)(getTickCount() - begin) / getTickFrequency() << "\n";
	return true;
}
#ifdef __cplusplus
extern "C"
{

#endif
LIBEXPORT unsigned char*LoadFrame(uchar* fisrtFrame, uchar* currentFrame, int height, int width, int channels111) {
	//std::this_thread::sleep_for(std::chrono::seconds(30));
	Mat fisrt = Mat(Size(width, height), CV_8UC1, Scalar(0));
	fisrt.data = fisrtFrame;

	Mat current = Mat(Size(width, height), CV_8UC1, Scalar(0));
	current.data = currentFrame;
	printf("row:%d,col:%d\n", fisrt.rows, fisrt.cols);
//	for (int i = 0; i < pre.rows; i++) {
//	for (int j = 0; j < pre.cols; j++) {
//		//int bgr = pre.at<int>(i, j);
//		uchar x = pre.at<uchar>(i, j);
//		printf("[%d][%d]:%d\n", i, j, x);
//		//Vec bgr = pre.at<Vec>(i, j);
//		//printf("b = %d, g = %d, r = %d\n", bgr[0], bgr[1], bgr[2]);
//	}
//	printf("\n");
//}

	//std::string file_name = "F:\\img\\result_gray_first" + std::to_string(channels111) + ".jpg";
	//cv::imwrite(file_name, fisrt);

	//file_name = "F:\\img\\result_gray_current" + std::to_string(channels111) + ".jpg";
	//cv::imwrite(file_name, current);

	Mat homography = calcHMat1(fisrt, current, 1200, 1.2, 1);
	cout << "M= " << endl << " " << homography << endl << endl;
	int total = homography.total();
	int elemSize = homography.elemSize();
	int len = homography.total() * homography.elemSize();

	uchar* buffer = (uchar*)malloc(len);
	memcpy(buffer, homography.data, len);
	return buffer;
//cv::imwrite("F:\\2.jpg", current);
	//cv::imwrite("F:\\1.jpg", pre);

	//cv::Mat src(width, height, CV_8UC1, currentFrame);	//cv::Mat dst;	//Canny(src, dst, 100, 200);

	//Mat tmp1 = cv::imread("F:\\1.jpg");
	//Mat tmp2 = cv::imread("F:\\2.jpg");
	//Mat homography = calcHMat1(tmp1, tmp2, 1200, 1.2, 1);



	






	// at 接口遍历 *****************    推荐使用
	//for (int i = 0; i < homography.rows; i++) {
	//	for (int j = 0; j < homography.cols; j++) {
	//		Vec3b bgr = homography.at<Vec3b>(i, j);
	//		printf("b = %d, g = %d, r = %d\n", bgr[0], bgr[1], bgr[2]);
	//	}
	//}




	//cout << "M= " << endl << " " << homography << endl << endl;
	//Mat* kk = new Mat;
	//*kk = homography;
	//uchar *s = homography.data;
	////int length = strlen((char*)homography.data);
	//int total = homography.total();
	//int elemSize = homography.elemSize();
	//int len = homography.total() * homography.elemSize();

	//uchar* buffer = (uchar*)malloc(len);
	//memcpy(buffer, homography.data, len);
	//return buffer;
}
LIBEXPORT  void ReleaseFrame(unsigned char* data) {
	if (data) {
		free(data);
	}
	
}
LIBEXPORT bool TestVideoShake(char* intputfilepath,char* outputfilepath) {
	int currentFrame = -1;

	VideoCapture inputVideo(intputfilepath);
	if (!inputVideo.isOpened())
	{
		std::cout << "Error opening video stream" << "\n";
		return false;
	}

	Mat frame;
	shared_ptr<Mat> homography_ptr;
	int ex = static_cast<int>(inputVideo.get(CAP_PROP_FOURCC));     // Get Codec Type- Int form
	double fps = inputVideo.get(CAP_PROP_FPS);
	Size size = Size(inputVideo.get(CAP_PROP_FRAME_WIDTH), inputVideo.get(CAP_PROP_FRAME_HEIGHT));

	String strintputfilepath = String(intputfilepath);
	int dot_pos = strintputfilepath.rfind(".");

	int total_frames = inputVideo.get(CAP_PROP_FRAME_COUNT);

	string output_file = strintputfilepath.substr(0, dot_pos) + "_stable.avi";
	cout << output_file << "\n";
	ex = VideoWriter::fourcc('X', 'V', 'I', 'D');
	cv::VideoWriter outputVideo(output_file, ex, fps, size);

	float unit_mat[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
	Mat lastHomography(3, 3, CV_32F, unit_mat);
	Mat homography;

	//handle the first frame
	vector<KeyPoint> keypoints_1;
	Mat gray_image1, gray_image2;

	int64 begin = getTickCount();
	int64 lastFrame = getTickCount();

	Mat image_1, image_2;
	while (1)
	{
		int64 iter_start = getTickCount();

		inputVideo >> frame;
		if (frame.empty())
		{
			break;
		}
		currentFrame++;

		if (frame.size().height == 0)
		{
			std::cout << "Read video failed." << "\n";
			break;
		}

		if (currentFrame == 0)
		{
			image_1 = frame.clone();
			std::cout << "\n";
			cvtColor(image_1, gray_image1, cv::COLOR_BGR2GRAY);
		}
		else
		{
			Mat forecast;
			image_2 = frame.clone();
			warpPerspective(image_2, forecast, lastHomography, size);
			cvtColor(forecast, gray_image2, cv::COLOR_BGR2GRAY);

			int64 calc_start = getTickCount();
			homography = calcHMat1(gray_image1, gray_image2, 1200, 1.2, 1);
			int64 calc_finish = getTickCount();
			////校正
			Mat warpImage2;
			warpPerspective(forecast, warpImage2, homography, size);
			outputVideo.write(warpImage2);
			int64 afterWriteFrame = getTickCount();

			Mat tmp;
			tmp = homography * lastHomography;
			lastHomography = tmp;

			if (currentFrame % 1000 == 0)
			{
				cout << currentFrame << endl;
				cout << float(currentFrame) / float(total_frames) << "\n";
			}

			if (currentFrame % 10 == 0)
			{
				cout << currentFrame << endl;
			}

		}
	}
	inputVideo.release();
	outputVideo.release();
	std::cout << "\n" << "Total time cost :" << (double)(getTickCount() - begin) / getTickFrequency() << "\n";
	return false;
}

LIBEXPORT void showNdarray(int* data, int rows, int cols) {
	std::this_thread::sleep_for(std::chrono::seconds(30));
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			printf("data[%d][%d] = %d\n", i, j, data[i * rows + j]);
		}
	}
}

LIBEXPORT uchar *get_mat_and_return_uchar(uchar *matrix, int rows, int cols, int channels)
{
	MatSO td;
	return td.get_mat_and_return_uchar(matrix, rows, cols, channels);
}

LIBEXPORT uchar *get_mat_and_return_buffer(uchar *matrix, int rows, int cols, int channels)
{
	MatSO td;
	return td.get_mat_and_return_buffer(matrix, rows, cols, channels);
}

#ifdef __cplusplus
}
#endif