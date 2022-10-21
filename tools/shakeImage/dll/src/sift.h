#ifndef _SIFT_H_
#define _SIFT_H_

#include<iostream>
#include<opencv2\core\core.hpp>
#include<opencv2\features2d\features2d.hpp>

using namespace std;
using namespace cv;

/*************************���峣��*****************************/

//��˹�˴�С�ͱ�׼���ϵ��size=2*(GAUSS_KERNEL_RATIO*sigma)+1,��������GAUSS_KERNEL_RATIO=2-3֮��
const double GAUSS_KERNEL_RATIO = 3;

//const int MAX_OCTAVES = 8;//�������������
const int MAX_OCTAVES = 2;//�������������

const float CONTR_THR = 0.04f;//Ĭ���ǵĶԱȶ���ֵ(D(x))

const float CURV_THR = 10.0f;//�ؼ�����������ֵ

const float INIT_SIGMA = 0.5f;//����ͼ��ĳ�ʼ�߶�

const int IMG_BORDER = 2;//ͼ��߽���ԵĿ��

const int MAX_INTERP_STEPS = 5;//�ؼ��㾫ȷ��ֵ����

const int ORI_HIST_BINS = 36;//���������㷽��ֱ��ͼ��BINS����

const float ORI_SIG_FCTR = 1.5f;//����������������ʱ�򣬸�˹���ڵı�׼������

const float ORI_RADIUS = 3 * ORI_SIG_FCTR;//����������������ʱ�����ڰ뾶����

const float ORI_PEAK_RATIO = 0.8f;//����������������ʱ��ֱ��ͼ�ķ�ֵ��

const int DESCR_WIDTH = 4;//������ֱ��ͼ�������С(4x4)
//const int DESCR_WIDTH = 2;//������ֱ��ͼ�������С(4x4)

const int DESCR_HIST_BINS = 8;//ÿ��������ֱ��ͼ�Ƕȷ����ά��

const float DESCR_MAG_THR = 0.2f;//�����ӷ�����ֵ

const float DESCR_SCL_FCTR = 3.0f;//����������ʱ��ÿ������Ĵ�С����




/************************sift��*******************************/
class MySift
{

public:
	//Ĭ�Ϲ��캯��
	MySift(int nfeatures = 0, int nOctaveLayers = 3, double contrastThreshold = 0.04,
		double edgeThreshold = 10, double sigma = 1.6, bool double_size = true) :nfeatures(nfeatures),
		nOctaveLayers(nOctaveLayers), contrastThreshold(contrastThreshold),
		edgeThreshold(edgeThreshold), sigma(sigma), double_size(double_size){}

	//��ó߶ȿռ�ÿ���м����
	int get_nOctave_layers() const { return nOctaveLayers; }

	//���ͼ��߶��Ƿ�����һ��
	bool get_double_size() const { return double_size; }

	//�������������
	int num_octaves(const Mat &image) const;

	//���ɸ�˹��������һ�飬��һ��ͼ��
	void create_initial_image(const Mat &image, Mat &init_image) const;

	//������˹������
	void build_gaussian_pyramid(const Mat &init_image, vector<vector<Mat>> &gauss_pyramid, int nOctaves) const;

	//������˹��ֽ�����
	void build_dog_pyramid(vector<vector<Mat>> &dog_pyramid, const vector<vector<Mat>> &gauss_pyramid) const;

	//DOG��������������
	void find_scale_space_extrema(const vector<vector<Mat>> &dog_pyr, const vector<vector<Mat>> &gauss_pyr,
		vector<KeyPoint> &keypoints) const;

	//�����������������
	void calc_descriptors(const vector<vector<Mat>> &dog_pyr, vector<KeyPoint> &keypoints,
		Mat &descriptors) const;

	//��������
	void detect(const Mat &image, vector<vector<Mat>> &gauss_pyr, vector<vector<Mat>> &dog_pyr,
		vector<KeyPoint> &keypoints) const;

	//����������
	void comput_des(const vector<vector<Mat>> &gauss_pyr, vector<KeyPoint> &keypoints, Mat &descriptors) const;


private:
	int nfeatures;//�趨����������ĸ���ֵ,�����ֵ����Ϊ0����Ӱ����
	int nOctaveLayers;//ÿ��������м����
	double contrastThreshold;//�Աȶ���ֵ��D(x)��
	double edgeThreshold;//�������Ե������ֵ
	double sigma;//��˹�߶ȿռ��ʼ��ĳ߶�
	bool double_size;//�Ƿ��ϲ���ԭʼͼ��

};//ע��������ķֺ�

#endif