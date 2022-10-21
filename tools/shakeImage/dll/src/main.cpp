#include"sift.h"
#include"display.h"
#include"match.h"
#include "ORBExtractor.h"

#include<opencv2\highgui\highgui.hpp>
#include<opencv2\calib3d\calib3d.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/opencv_modules.hpp>
using namespace cv;
using namespace std;



#include<fstream>
#include<stdlib.h>
#include<direct.h>
#include<iomanip>
#include<filesystem>

int image_idx = 0;
Mat descriptors_1;

Mat logEnhance(Mat image_1)
{

	Mat imageLog(image_1.size(), CV_32FC3);
	for (int i = 0; i < image_1.rows; i++)
	{
		for (int j = 0; j < image_1.cols; j++)
		{
			imageLog.at<Vec3f>(i, j)[0] = log(1 + image_1.at<Vec3b>(i, j)[0]);
			imageLog.at<Vec3f>(i, j)[1] = log(1 + image_1.at<Vec3b>(i, j)[1]);
			imageLog.at<Vec3f>(i, j)[2] = log(1 + image_1.at<Vec3b>(i, j)[2]);
		}
	}
	//归一化到0~255  
	normalize(imageLog, imageLog, 0, 255, NORM_MINMAX);
	//转换成8bit图像显示  
	convertScaleAbs(imageLog, imageLog);
	return imageLog;
};

namespace ACE {
	//Gray
	Mat stretchImage(Mat src) {
		int row = src.rows;
		int col = src.cols;
		Mat dst(row, col, CV_64FC1);
		double MaxValue = 0;
		double MinValue = 256.0;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				MaxValue = max(MaxValue, src.at<double>(i, j));
				MinValue = min(MinValue, src.at<double>(i, j));
			}
		}
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				dst.at<double>(i, j) = (1.0 * src.at<double>(i, j) - MinValue) / (MaxValue - MinValue);
				if (dst.at<double>(i, j) > 1.0) {
					dst.at<double>(i, j) = 1.0;
				}
				else if (dst.at<double>(i, j) < 0) {
					dst.at<double>(i, j) = 0;
				}
			}
		}
		return dst;
	}

	Mat getPara(int radius) {
		int size = radius * 2 + 1;
		Mat dst(size, size, CV_64FC1);
		for (int i = -radius; i <= radius; i++) {
			for (int j = -radius; j <= radius; j++) {
				if (i == 0 && j == 0) {
					dst.at<double>(i + radius, j + radius) = 0;
				}
				else {
					dst.at<double>(i + radius, j + radius) = 1.0 / sqrt(i * i + j * j);
				}
			}
		}
		double sum = 0;
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				sum += dst.at<double>(i, j);
			}
		}
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				dst.at<double>(i, j) = dst.at<double>(i, j) / sum;
			}
		}
		return dst;
	}

	Mat NormalACE(Mat src, int ratio, int radius) {
		Mat para = getPara(radius);
		int row = src.rows;
		int col = src.cols;
		int size = 2 * radius + 1;
		Mat Z(row + 2 * radius, col + 2 * radius, CV_64FC1);
		for (int i = 0; i < Z.rows; i++) {
			for (int j = 0; j < Z.cols; j++) {
				if ((i - radius >= 0) && (i - radius < row) && (j - radius >= 0) && (j - radius < col)) {
					Z.at<double>(i, j) = src.at<double>(i - radius, j - radius);
				}
				else {
					Z.at<double>(i, j) = 0;
				}
			}
		}

		Mat dst(row, col, CV_64FC1);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				dst.at<double>(i, j) = 0.f;
			}
		}
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				if (para.at<double>(i, j) == 0) continue;
				for (int x = 0; x < row; x++) {
					for (int y = 0; y < col; y++) {
						double sub = src.at<double>(x, y) - Z.at<double>(x + i, y + j);
						double tmp = sub * ratio;
						if (tmp > 1.0) tmp = 1.0;
						if (tmp < -1.0) tmp = -1.0;
						dst.at<double>(x, y) += tmp * para.at<double>(i, j);
					}
				}
			}
		}
		return dst;
	}

	Mat FastACE(Mat src, int ratio, int radius) {
		int row = src.rows;
		int col = src.cols;
		if (min(row, col) <= 2) {
			Mat dst(row, col, CV_64FC1);
			for (int i = 0; i < row; i++) {
				for (int j = 0; j < col; j++) {
					dst.at<double>(i, j) = 0.5;
				}
			}
			return dst;
		}

		Mat Rs((row + 1) / 2, (col + 1) / 2, CV_64FC1);

		resize(src, Rs, Size((col + 1) / 2, (row + 1) / 2));
		Mat Rf = FastACE(Rs, ratio, radius);
		resize(Rf, Rf, Size(col, row));
		resize(Rs, Rs, Size(col, row));
		Mat dst(row, col, CV_64FC1);
		Mat dst1 = NormalACE(src, ratio, radius);
		Mat dst2 = NormalACE(Rs, ratio, radius);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				dst.at<double>(i, j) = Rf.at<double>(i, j) + dst1.at<double>(i, j) - dst2.at<double>(i, j);
			}
		}
		return dst;
	}

	Mat getACE(Mat src, int ratio, int radius) {
		int row = src.rows;
		int col = src.cols;
		vector <Mat> v;
		split(src, v);
		v[0].convertTo(v[0], CV_64FC1);
		v[1].convertTo(v[1], CV_64FC1);
		v[2].convertTo(v[2], CV_64FC1);
		Mat src1(row, col, CV_64FC1);
		Mat src2(row, col, CV_64FC1);
		Mat src3(row, col, CV_64FC1);

		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				src1.at<double>(i, j) = 1.0 * src.at<Vec3b>(i, j)[0] / 255.0;
				src2.at<double>(i, j) = 1.0 * src.at<Vec3b>(i, j)[1] / 255.0;
				src3.at<double>(i, j) = 1.0 * src.at<Vec3b>(i, j)[2] / 255.0;
			}
		}
		src1 = stretchImage(FastACE(src1, ratio, radius));
		src2 = stretchImage(FastACE(src2, ratio, radius));
		src3 = stretchImage(FastACE(src3, ratio, radius));

		Mat dst1(row, col, CV_8UC1);
		Mat dst2(row, col, CV_8UC1);
		Mat dst3(row, col, CV_8UC1);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				dst1.at<uchar>(i, j) = (int)(src1.at<double>(i, j) * 255);
				if (dst1.at<uchar>(i, j) > 255) dst1.at<uchar>(i, j) = 255;
				else if (dst1.at<uchar>(i, j) < 0) dst1.at<uchar>(i, j) = 0;
				dst2.at<uchar>(i, j) = (int)(src2.at<double>(i, j) * 255);
				if (dst2.at<uchar>(i, j) > 255) dst2.at<uchar>(i, j) = 255;
				else if (dst2.at<uchar>(i, j) < 0) dst2.at<uchar>(i, j) = 0;
				dst3.at<uchar>(i, j) = (int)(src3.at<double>(i, j) * 255);
				if (dst3.at<uchar>(i, j) > 255) dst3.at<uchar>(i, j) = 255;
				else if (dst3.at<uchar>(i, j) < 0) dst3.at<uchar>(i, j) = 0;
			}
		}
		vector <Mat> out;
		out.push_back(dst1);
		out.push_back(dst2);
		out.push_back(dst3);
		Mat dst;
		merge(out, dst);
		return dst;
	}
}
using namespace ACE;


void filterKPbyMatch(const Mat& image1, const Mat& image2, const vector<KeyPoint>& keys_1, const vector<KeyPoint>& keys_2, vector<vector<DMatch>>& dmatches)
{
	vector<vector<DMatch>> left_matchs;
	Mat mask1 = imread("1_mask.png", 0);
	Mat mask2 = imread("300_mask.png", 0);
	for (int i = 0; i < dmatches.size(); i++)
	{
		int x1 = keys_1[dmatches[i][0].queryIdx].pt.x;
		int y1 = keys_1[dmatches[i][0].queryIdx].pt.y;

		int x2 = keys_2[dmatches[i][0].trainIdx].pt.x;
		int y2 = keys_2[dmatches[i][0].trainIdx].pt.y;

		if (mask1.at<uchar>(y1, x1) == 0 || mask2.at<uchar>(y2, x2) == 0)
		{
			//dmatches.erase(dmatches.begin() + i);
		}
		else
		{
			left_matchs.push_back(dmatches[i]);
		}
	}
	dmatches.clear();
	dmatches = left_matchs;
}

void filterKPbyMatch(const vector<KeyPoint>& keys_1, vector<vector<DMatch>>& dmatches)
{
	vector<vector<DMatch>> left_matchs;
	Mat mask1 = imread("1_mask.png", 0);
	if (mask1.size().width == 0)
	{
		return;
	}
	for (int i = 0; i < dmatches.size(); i++)
	{
		int x1 = keys_1[dmatches[i][0].queryIdx].pt.x;
		int y1 = keys_1[dmatches[i][0].queryIdx].pt.y;

		if (mask1.at<uchar>(y1, x1) == 0)
		{
			//dmatches.erase(dmatches.begin() + i);
		}
		else
		{
			left_matchs.push_back(dmatches[i]);
		}
	}
	dmatches.clear();
	dmatches = left_matchs;
}

template<typename T>
void findHistogram(vector<T> vals, T max)
{
	int fsize = vals.size();
	if (max >= 100)
	{
		cout <<"Max is larger than 20, don't show histogram." << max << "\n";
		return;
	}
	vector<int> freqs(100, 0);

	//compute frequencies
	for (int i = 0; i < vals.size(); i++)
		freqs[(int)floor(vals[i])]++;

	//print histogram
	cout << "\n....Histogram....,total distance size = " << fsize << "\n";

	for (int i = 0; i < 100; i++) {
		cout << left;
		cout << setw(5) << i;
		cout << setw(5) << freqs[i] << "\n";
	}
}

void GammaCorrection(cv::Mat& dst_image, const cv::Mat& src_image, const int& constant, const float& gamma)
{
	// step1：check the input parameter : 检查输入参数
	assert(constant > 0);
	assert(gamma > 0);
	assert(src_image.size == dst_image.size);

	// step2: traverse every pixel and execute gamma correction: 遍历每个像素点并且执行伽马矫正
	for (size_t i = 0; i < src_image.rows; i++)
	{
		for (size_t j = 0; j < src_image.cols; j++)
		{
			for (size_t k = 0; k < 3; k++)
			{
				// dst = c * src ^ r ( r > 0)
				auto result = constant * std::pow((float)src_image.at<cv::Vec3b>(i, j)[k] / 255.0, gamma) * 255;

				// prevent grayscale value from overflowing: 防止灰度值溢出
				dst_image.at<cv::Vec3b>(i, j)[k] = cv::saturate_cast<uchar>(result);
			}
		}
	}
}

//void GetGammaCorrection(Mat& src, Mat& dst, const float fGamma)
//{
//	unsigned char bin[256];
//	for (int i = 0; i < 256; ++i)
//	{
//		bin[i] = cv::saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
//	}
//	dst = src.clone();
//	const int channels = dst.channels();
//	switch (channels)
//	{
//	case 1:
//	{
//		cv::MatIterator_ it;
//
//		for (it = dst.begin(), end = dst.end(); it != end; it++) *it = bin[(*it)];
//		break;
//	}
//	case 3:
//	{
//		cv::MatIterator_ it, end;
//		for (it = dst.begin(), end = dst.end(); it != end; it++)
//		{
//			(*it)[0] = bin[((*it)[0])]; (*it)[1] = bin[((*it)[1])]; (*it)[2] = bin[((*it)[2])];
//		}
//		break;
//	}
//	}
//}

Mat calcHMat(Mat& image_1, Mat& image_2,
	int nFeatures, float scaleFactor, int nLevels, int iniThFAST, int minThFAST)
{
	string change_model = string("perspective");
	//Mat image_1, image_2;

	//创建文件夹保存图像
	char* newfile = ".\\image_save";
	_mkdir(newfile);

	//待配准图像特征点检测和描述
	vector<vector<Mat>> gauss_pyr_2, dog_pyr_2;
	vector<KeyPoint> keypoints_1;
	vector<KeyPoint> keypoints_2;
	Mat descriptors_2;

	Mat image1, image2;

	//算法运行总时间开始计时
	double total_count_begin = (double)getTickCount();

	ORB_SLAM2::ORBextractor orb(nFeatures, scaleFactor, nLevels, iniThFAST, minThFAST);
	orb(image_1, Mat(), keypoints_1, descriptors_1, 0);

	ORB_SLAM2::ORBextractor orb2(nFeatures, scaleFactor, nLevels, iniThFAST, minThFAST);
	orb2(image_2, Mat(), keypoints_2, descriptors_2, 0);


	//最近邻与次近邻距离比匹配
	double before_knn_match = (double)getTickCount();

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

	//Ptr<DescriptorMatcher> matcher;
	//
	//matcher->knnMatch(pos1, descriptors_2, dmatches, 10);
	//std::vector<vector<DMatch>> dmatches;
	//Ptr<DescriptorMatcher> matcher = new FlannBasedMatcher;
	//matcher = new BFMatcher(NORM_HAMMING);
	//matcher->knnMatch(descriptors_1, descriptors_2, dmatches, 8);


	std::vector<vector<DMatch>> dmatches_pose;
	Ptr<DescriptorMatcher> matcher_pos = new FlannBasedMatcher;
	matcher_pos = new BFMatcher(NORM_L2);
	matcher_pos->knnMatch(pos1, pos2, dmatches_pose, 10);

//#ifdef _DEBUG
//	cout << "根据距离匹配的的关键点对： " << dmatches.size() << "\n";
//#endif 


	double after_knn_match = (double)getTickCount();
	//match_des(descriptors_1, descriptors_2, dmatches, COS);
	//filterKPbyMatch(keypoints_1, dmatches);
	// 不做mask
	//filterKPbyMatch(keypoints_1, dmatches_pose);

	//vector<vector<DMatch>> mix_matches;
	//for (int i = 0; i < dmatches_pose.size(); i++)
	//{	
	//	vector<DMatch> pos_match = dmatches_pose.at(i);
	//	for (int j = 0; j < dmatches.size(); j++)
	//	{
	//		vector<DMatch> des_match = dmatches.at(j);
	//		if (pos_match[0].queryIdx == des_match[0].queryIdx)
	//		{

	//			for (int k = 0; k < 4; k++)
	//			{
	//				if (pos_match[k].trainIdx == des_match[0].trainIdx)
	//				{
	//					mix_matches.push_back(des_match);
	//				}
	//			}
	//		}
	//	}
	//}


//#ifdef _DEBUG
	//cout << "\n knn match的匹配数： " << dmatches_pose.size() << "\n";
/*#endif*/ 


	Mat matched_lines;
	vector<DMatch> right_matches;
	for (size_t i = 0; i < dmatches_pose.size(); i++)
	{
		right_matches.push_back(dmatches_pose[i][0]);//直接保存的dmatches_pos	
	}
	//vector<Point2f> seqKeyPoints1, seqKeyPoints2;
	//for (int i = 0; i < right_matches.size(); i++)
	//{
	//	seqKeyPoints1.push_back(keypoints_1.at(right_matches.at(i).queryIdx).pt);
	//	seqKeyPoints2.push_back(keypoints_2.at(right_matches.at(i).trainIdx).pt);
	//}
	//Mat homography1 = findHomography(seqKeyPoints1, seqKeyPoints2, 0);

	//drawMatches(image_1, keypoints_1, image_2, keypoints_2, right_matches, matched_lines, Scalar(0, 255, 0), Scalar(0, 0, 255),
	//	vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//cv::imwrite(".\\image_save\\before_ransac_正确匹配点对" + to_string(++image_idx) + ".jpg", matched_lines);

	//right_matches.clear();


	//Mat homography1 = Mat();
	//homography1 = findHomography(keypoints_1, keypoints_2, 0);


	//求解Homography矩阵
	double after_h_match = (double)getTickCount();
	Mat homography = match(dmatches_pose, keypoints_1, keypoints_2, change_model,
		right_matches, after_h_match);

	////显示统计信息
	//double cost1 = (before_knn_match - total_count_begin) / getTickFrequency();
	//std::cout << "\n" << "提取orb： " << cost1 << "s";
	//double knn_match_cost = (after_knn_match - before_knn_match) / getTickFrequency();
	//std::cout << "\n" << "临近匹配时间： " << knn_match_cost << "s";
	//double h_match_cost = (after_h_match - after_knn_match) / getTickFrequency();
	//std::cout << "\n" << "求矩阵时间是： " << h_match_cost << "s";
	//double total_time = (after_h_match - total_count_begin) / getTickFrequency();
	//std::cout << "\n" << "总配准时间是： " << total_time << "s" << "\n";

	//std::cout << "\n" << "变换矩阵是：" << "\n";
	//std::cout << homography << "\n";

	//绘制正确匹配点对连线图
	//vector<char> rightMatchMask;
	//for (int i = 0; i < right_matches.size(); i++)
	//{
	//	rightMatchMask.push_back(0);
	//}
	//drawMatches(image_1, keypoints_1, image_2, keypoints_2, right_matches, matched_lines, Scalar(0, 255, 0), Scalar(0, 0, 255),
	//	vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//cv::imwrite(".\\image_save\\正确匹配点对"+ to_string(++image_idx) + ".jpg", matched_lines);
	//Mat dst;
	//cv::resize(matched_lines, dst, Size(), 0.25, 0.25);
	//cv::imshow("right matches", dst);
	//cv::waitKey();

#ifdef _DEBUG

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
	cv::imwrite(".\\image_save\\正确匹配点对"+ to_string(image_idx) + ".jpg", matched_lines);

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

//#include <filesystem>
int gcurrentFrame = -1;
bool test(double width,double height,Mat& frame) {

	shared_ptr<Mat> homography_ptr;



	float unit_mat[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
	Mat lastHomography(3, 3, CV_32F, unit_mat);
	Mat homography;

	//handle the first frame
	vector<KeyPoint> keypoints_1;
	Mat gray_image1, gray_image2;

	int64 begin = getTickCount();
	int64 lastFrame = getTickCount();

	Size size = Size(width, height);
	//while (1)
	{
		Mat image_1, image_2;
		int64 iter_start = getTickCount();


		gcurrentFrame++;

		if (frame.size().height == 0)
		{
			std::cout << "Read video failed." << "\n";
			return false;
		}

		if (gcurrentFrame == 0)
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
			//////校正
			//Mat warpImage2;
			//warpPerspective(forecast, warpImage2, homography, size);
			//outputVideo.write(warpImage2);
//			int64 afterWriteFrame = getTickCount();

			Mat tmp;
			tmp = homography * lastHomography;
			lastHomography = tmp;
		}
	}
	
	//outputVideo.release();
	std::cout << "\n" << "Total time cost :" << (double)(getTickCount() - begin) / getTickFrequency() << "\n";
	return true;
}
int main(int argc, char *argv[])
{
	String dirPath = ".\\image_save";
	//if (std::filesystem::exists(dirPath))
	//{
	//	for (const auto&entry : std::filesystem::directory_iterator(dirPath))
	//		std::filesystem::remove_all(entry.path());
	//}

	String arg1 = String(argv[1]);
	String arg2 = String(argv[2]);
	Mat image_1, image_2, frame;

	const int iniThFAST = 60;
	const int minThFAST = 60;
	const int featuresAmout = 1200;

	if (argc != 3)
	{
		std::cout << "SIFT_Registration.exe -v [video_file_full_path]" << "\n";
		return -1;
	}

    if (arg1 == "-v" && argc == 3)
	{
		int currentFrame = -1;

		VideoCapture inputVideo(arg2);
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
		int dot_pos = arg2.rfind(".");
		//string output_file = arg2.substr(0, dot_pos) + "_stable" + arg2.substr(dot_pos, arg2.size()-1);
		//cout << output_file << "\n";
		//cv::VideoWriter outputVideo(output_file, ex, fps, size);
		int total_frames = inputVideo.get(CAP_PROP_FRAME_COUNT);

		string output_file = arg2.substr(0, dot_pos) + "_stable.avi";
		cout << output_file << "\n";
		ex = VideoWriter::fourcc('X', 'V', 'I', 'D');
		cv::VideoWriter outputVideo(output_file, ex, fps, size);

		float unit_mat[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
		Mat lastHomography(3,3, CV_32F, unit_mat);
		Mat homography;

		//handle the first frame
		vector<KeyPoint> keypoints_1;
		Mat gray_image1, gray_image2;

		int64 begin = getTickCount();
		int64 lastFrame = getTickCount();
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
		std::cout << "\n"<<"Total time cost :" << (double)(getTickCount() - begin) / getTickFrequency() << "\n";
	}else if(arg1 == "-v" && argc == 5)
	{
		int currentFrame = -1;

		VideoCapture inputVideo(arg2);
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
		cv::VideoWriter outputVideo("stable_" + arg2, ex, fps, size);
		float unit_mat[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
		Mat lastHomography(3, 3, CV_32F, unit_mat);
		Mat homography;

		//handle the first frame
		vector<KeyPoint> keypoints_1;
		Mat gray_image1;

		int64 begin = getTickCount();
		int64 lastFrame = getTickCount();
		while (1)
		{
			int64 iter_start = getTickCount();

			inputVideo >> frame;
			currentFrame++;

			if (frame.size().height == 0)
			{
				std::cout << "Read video failed." << "\n";
				break;
			}

			int arg3 = stoi(String(argv[3]));
			int arg4 = stoi(String(argv[4]));
			if (currentFrame < arg3)
			{
				continue;
			} else if (currentFrame == arg3)
			{
				std::cout << "current index = " << currentFrame;
				image_1 = frame.clone();
				std::cout << "\n";

				//参考图像特征点检测和描述
				vector<vector<Mat>> gauss_pyr_1, dog_pyr_1;
				Mat descriptors_1;

				cvtColor(image_1, gray_image1, cv::COLOR_BGR2GRAY);
				ORB_SLAM2::ORBextractor orb(featuresAmout, 1.2, 1, iniThFAST, minThFAST);
				orb(gray_image1, Mat(), keypoints_1, descriptors_1, 0);
			}
			else if (currentFrame > arg4)
			{
				break;
			} 
			else if (currentFrame > arg3 && currentFrame <= arg4)
			{
				std::cout << "current index = " << currentFrame;
				Mat forecast;
				image_2 = frame.clone();
				warpPerspective(image_2, forecast, lastHomography, size);				

				int64 calc_start = getTickCount();
				homography = calcHMat(image_1, forecast, featuresAmout, 1.2, 1, iniThFAST, minThFAST);
				int64 calc_finish = getTickCount();
				////校正
				Mat warpImage2;
				warpPerspective(forecast, warpImage2, homography, size);
				outputVideo.write(warpImage2);
				int64 afterWriteFrame = getTickCount();

				Mat tmp;
				tmp = homography * lastHomography;
				lastHomography = tmp;

				int64 iterEnd = getTickCount();
				std::cout << "before calcH :" << (double)(calc_start - iter_start) / getTickFrequency();
				std::cout << ", calcH :" << (double)(calc_finish - calc_start) / getTickFrequency();
				std::cout << ", writeframe:" << (double)(afterWriteFrame - calc_finish) / getTickFrequency();
				std::cout << ", total of iter:" << (double)(iterEnd - iter_start) / getTickFrequency() << "\n";

			}
		}
		inputVideo.release();
		outputVideo.release();
		std::cout << "\n" << "Total time cost :" << (double)(getTickCount() - begin) / getTickFrequency() << "\n";

	}
	//else if (arg1 == "-vi" && argc == 8)
	//{
	//	int currentFrame = -1;

	//	VideoCapture inputVideo(arg2);
	//	if (!inputVideo.isOpened())
	//	{
	//		std::cout << "Error opening video stream" << "\n";
	//		return false;
	//	}

	//	int arg3 = std::stoi(String(argv[3]));
	//	int arg4 = std::stoi(String(argv[4]));
	//	int arg5 = std::stoi(String(argv[5]));
	//	float arg6 = std::stof(String(argv[6]));
	//	int arg7 = std::stoi(String(argv[7]));

	//	shared_ptr<Mat> homography_ptr;
	//	while (1)
	//	{
	//		inputVideo >> frame;
	//		++currentFrame;

	//		if (currentFrame == arg3)
	//			image_1 = frame.clone();
	//		if (currentFrame == arg4)
	//		{
	//			image_2 = frame.clone();
	//			break;
	//		}
	//	}
	//	cout << "Get the frame from video, current frame =" << currentFrame << "\n";

	//	Mat homography = calcHMat(image_1, image_2, arg5, arg6, arg7);

	//	//校正
	//	Mat fusion_image, mosaic_image, regist_image;
	//	image_fusion(image_1, image_2, homography, fusion_image, mosaic_image, regist_image);
	//	String image_name = ".\\image_save\\融合后的镶嵌图像" + to_string(currentFrame) + ".jpg";
	//	cv::imwrite(image_name, mosaic_image);

	//	inputVideo.release();
	//}
	//else if (arg1 == "-base" && argc == 7)
	//{
	//	String arg2 = argv[2];
	//	String arg3 = argv[3];
	//	int featuresAmout = std::stoi(String(argv[4]));
	//	float scale_factor = std::stof(String(argv[5]));
	//	int nlevel = std::stoi(String(argv[6]));

	//	Mat image_1 = imread(arg2, 1);
	//	//参考图像特征点检测和描述
	//	vector<vector<Mat>> gauss_pyr_1, dog_pyr_1;
	//	Mat descriptors_1;



	//	VideoCapture inputVideo(arg3);
	//	if (!inputVideo.isOpened())
	//	{
	//		std::cout << "Error opening video stream" << "\n";
	//		return false;
	//	}

	//	//cv::Rect crop_region_video(859, 509, 1420, 1166);
	//	//cv::Rect crop_region_im(1083, 924, 1085, 883);


	//	cv::Rect crop_region_video(582, 478, 503, 427);

	//	Mat frame;
	//	inputVideo >> frame;
	//	Mat image_2 = frame.clone();

	//	//Mat cropped_1 = image_1.clone();
	//	//Mat cropped_2 = image_2.clone();
	//	//Mat cropped_1 = image_1(crop_region_im);
	//	//Mat cropped_2 = image_2(crop_region_video);
	//	Mat cropped_1 = image_1(crop_region_video);
	//	Mat cropped_2 = image_2(crop_region_video);

	//	Mat gray_img1, gray_img2;
	//	cvtColor(cropped_1, gray_img1, cv::COLOR_BGR2GRAY);
	//	cvtColor(cropped_2, gray_img2, cv::COLOR_BGR2GRAY);


	//	//handle the first frame
	//	vector<KeyPoint> keypoints_1;
	//	ORB_SLAM2::ORBextractor orb(featuresAmout, scale_factor, nlevel, iniThFAST, minThFAST);
	//	orb(input_image_1, Mat(), keypoints_1, descriptors_1, 0);

	//	//image_1 = image1.clone();
	//	//GammaCorrection(image1, image_1, 1, 1.5);
	//	imwrite(".\\image_save\\1.jpg", input_image_1);
	//	imwrite(".\\image_save\\2.jpg", gray_img2);

	//	Mat homography = calcHMat(input_image_1, gray_img2, keypoints_1, featuresAmout, scale_factor, nlevel, iniThFAST, minThFAST);
	//	int64 calc_finish = getTickCount();
	//	////校正
	//	Mat warpImage2;
	//	warpPerspective(image_2, warpImage2, homography, Size(image_2.size[0], image_2.size[1]));


	//	cv::imwrite(".\\image_save\\校正后的图像.jpg", warpImage2);

	//	//校正
	//	Mat fusion_image, mosaic_image, regist_image;
	//	image_fusion(image_1, image_2, homography, fusion_image, mosaic_image, regist_image);
	//	String image_name = ".\\image_save\\融合后的镶嵌图像.jpg";
	//	cv::imwrite(image_name, mosaic_image);
	//}
	else if (arg1 == "-compare_video" && argc == 7)
	{
		String arg2 = argv[2];
		String arg3 = argv[3];
		int featuresAmout = std::stoi(String(argv[4]));
		float scale_factor = std::stof(String(argv[5]));
		int nlevel = std::stoi(String(argv[6]));
	
		//参考图像特征点检测和描述
		vector<vector<Mat>> gauss_pyr_1, dog_pyr_1;
		Mat descriptors_1;
		
		VideoCapture inputVideo1(arg2);
		if (!inputVideo1.isOpened())
		{
			std::cout << "Error opening video stream" << "\n";
			return false;
		}
		inputVideo1 >> image_1;
		Size size = Size(inputVideo1.get(CAP_PROP_FRAME_WIDTH), inputVideo1.get(CAP_PROP_FRAME_HEIGHT));


		VideoCapture inputVideo2(arg3);
		if (!inputVideo2.isOpened())
		{
			std::cout << "Error opening video stream" << "\n";
			return false;
		}
		inputVideo2 >> image_2;

		//cv::Rect crop_region_video(859, 509, 1420, 1166);
		//cv::Rect crop_region_im(1083, 924, 1085, 883);
		cv::Rect crop_region_video(680, 470, 1598, 1272);

		Mat gray_image2;
		cvtColor(image_2, gray_image2, cv::COLOR_BGR2GRAY);


		Mat gray_image1;
		cvtColor(image_1, gray_image1, cv::COLOR_BGR2GRAY);

		//Mat cropped_1 = image_1.clone();
		//Mat cropped_2 = image_2.clone();
		//Mat cropped_1 = image_1(crop_region_im);
		//Mat cropped_2 = image_2(crop_region_video);
		Mat cropped_1 = gray_image1(crop_region_video);
		Mat cropped_2 = gray_image2(crop_region_video);


		//img_norm = src / 255.0  # 注意255.0得采用浮点数
		//img_gamma = np.power(img_norm, 1.5) * 255.0
		//dst = img_gamma.astype(np.uint8)

		Mat lookUpTable(1, 256, CV_8U);
		uchar* p = lookUpTable.ptr();
		for (int i = 0; i < 256; ++i)
			p[i] = saturate_cast<uchar>(pow(i / 255.0, 1.5) * 255.0);
		Mat input_image_1 = gray_image1.clone();
		LUT(gray_image1, lookUpTable, input_image_1);

		imwrite(".\\image_save\\1.jpg", gray_image1);
		imwrite(".\\image_save\\2.jpg", gray_image2);

		Mat homography = calcHMat(input_image_1, gray_image2, featuresAmout, scale_factor, nlevel, iniThFAST, minThFAST);
		int64 calc_finish = getTickCount();
		////校正
		Mat warpImage2;
		cout << image_2.size();
		warpPerspective(image_2, warpImage2, homography, size);


		cv::imwrite(".\\image_save\\校正后的图像.jpg", warpImage2);

		//校正
		Mat fusion_image, mosaic_image, regist_image;
		image_fusion(image_1, image_2, homography, fusion_image, mosaic_image, regist_image);
		String image_name = ".\\image_save\\融合后的镶嵌图像.jpg";
		cv::imwrite(image_name, mosaic_image);
		}


		cv::destroyAllWindows();
}

//#include <windows.h>
//#include "pch.h"

//bool APIENTRY DllMain(HMODULE hModule,
//	DWORD  ul_reason_for_call,
//	LPVOID lpReserved
//)
//{
//	switch (ul_reason_for_call)
//	{
//	case DLL_PROCESS_ATTACH:
//	case DLL_THREAD_ATTACH:
//	case DLL_THREAD_DETACH:
//	case DLL_PROCESS_DETACH:
//		break;
//	}
//	return TRUE;
//}