// Dll2.cpp: ���� DLL Ӧ�ó���ĵ���������
#define EXPORT __declspec(dllexport)
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// MatSO����
class MatSO
{
public:
	// �ú�������uchar���ݣ�תΪMat���ݣ���ʾ��Ȼ���ٷ���uchar����
	uchar *get_mat_and_return_uchar(uchar *matrix, int rows, int cols, int channels);

	// �ú�������uchar���ݣ�תΪMat���ݣ���ʾ��Ȼ���ٷ���buffle����
	uchar *get_mat_and_return_buffer(uchar *matrix, int rows, int cols, int channels);
};


// mat���ݽ��ղ���uchar����
uchar *MatSO::get_mat_and_return_uchar(uchar *matrix, int rows, int cols, int channels)
{
	// �����յ�ucharתΪMat
	cout << "rows=" << rows << "\ncols=" << cols << "\nchannels=" << channels;
	Mat input_mat = Mat(Size(cols, rows), CV_8UC3, Scalar(255, 255, 255));
	input_mat.data = matrix;

	// ��ʾMat
	imshow("input_mat", input_mat);
	cv::waitKey(0);

	// ��MatתΪuchar���Ͳ�����
	// ע�⣺Ҫ��ס����Mat��rol��row��channels����python�в�����ȷ����
	uchar *s = input_mat.data; // Matתucahr*
	return s;
}


// mat���ݽ��ղ���buffer����
uchar *MatSO::get_mat_and_return_buffer(uchar *matrix, int rows, int cols, int channels)
{
	// �����յ�ucharתΪMat
	cout << "rows=" << rows << "\ncols=" << cols << "\nchannels=" << channels;
	Mat input_mat = Mat(Size(cols, rows), CV_8UC3, Scalar(255, 255, 255));
	input_mat.data = matrix;

	// ��ʾMat
	imshow("input_mat", input_mat);
	cv::waitKey(0);

	// ��MatתΪbuffer������
	// ע�⣺Ҫ��ס����Mat��rol��row��channels����python�в�����ȷ����
	int height = input_mat.cols;
	int width = input_mat.rows;
	uchar *buffer = (uchar *)malloc(sizeof(uchar) * height * width * 3);
	memcpy(buffer, input_mat.data, height * width * 3);
	return buffer;
}

extern "C"
{
	MatSO td; // ��װMatSO��ʹ֮������so�ļ������

	uchar *get_mat_and_return_uchar(uchar *matrix, int rows, int cols, int channels)
	{
		return td.get_mat_and_return_uchar(matrix, rows, cols, channels);
	}

	uchar *get_mat_and_return_buffer(uchar *matrix, int rows, int cols, int channels)
	{
		return td.get_mat_and_return_buffer(matrix, rows, cols, channels);
	}
}

