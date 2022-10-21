// Dll2.cpp: 定义 DLL 应用程序的导出函数。
#define EXPORT __declspec(dllexport)
#include <opencv2/opencv.hpp>
#include <iostream>

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
	Mat input_mat = Mat(Size(cols, rows), CV_8UC3, Scalar(255, 255, 255));
	input_mat.data = matrix;

	// 显示Mat
	imshow("input_mat", input_mat);
	cv::waitKey(0);

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
	imshow("input_mat", input_mat);
	cv::waitKey(0);

	// 将Mat转为buffer并返回
	// 注意：要记住这里Mat的rol、row和channels，在python中才能正确接收
	int height = input_mat.cols;
	int width = input_mat.rows;
	uchar *buffer = (uchar *)malloc(sizeof(uchar) * height * width * 3);
	memcpy(buffer, input_mat.data, height * width * 3);
	return buffer;
}

extern "C"
{
	MatSO td; // 包装MatSO，使之可以在so文件外调用

	uchar *get_mat_and_return_uchar(uchar *matrix, int rows, int cols, int channels)
	{
		return td.get_mat_and_return_uchar(matrix, rows, cols, channels);
	}

	uchar *get_mat_and_return_buffer(uchar *matrix, int rows, int cols, int channels)
	{
		return td.get_mat_and_return_buffer(matrix, rows, cols, channels);
	}
}

