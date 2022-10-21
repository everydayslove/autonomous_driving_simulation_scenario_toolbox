#include"sift.h"

#include<string>
#include<opencv2\core\core.hpp>//opencv�������ݽṹ
#include<opencv2\highgui\highgui.hpp>//ͼ�����
#include<opencv2\imgproc\imgproc.hpp>//����ͼ������
#include<opencv2\features2d\features2d.hpp>//������ȡ
//#include<opencv2\contrib\contrib.hpp>

#include<iostream>//�������
#include<vector>//vector
#include<algorithm>
#include<math.h>
#define CV_StsBadArg -5

/******************��������ͼ���С�����˹������������****************************/
/*image��ʾԭʼ����Ҷ�ͼ��,inline��������������������
double_size_image��ʾ�Ƿ��ڹ���������֮ǰ�ϲ���ԭʼͼ��
*/

int MySift::num_octaves(const Mat &image) const
{
	int temp;
	float size_temp = (float)min(image.rows, image.cols);
	temp = cvRound(log(size_temp) / log((float)2) - 2);

	if (double_size)
		temp += 1;
	if (temp > MAX_OCTAVES)//�߶ȿռ������������ΪMAX_OCTAVES
		temp = MAX_OCTAVES;

	return temp;
}

/************************������˹��������һ�飬��һ��ͼ��************************************/
/*image��ʾ����ԭʼͼ��
 init_image��ʾ���ɵĸ�˹�߶ȿռ�ĵ�һ��ͼ��
 */
void MySift::create_initial_image(const Mat &image, Mat &init_image) const
{
	//ת����Ϊ�Ҷ�ͼ��
	Mat gray_image;
	if (image.channels() != 1)
		cvtColor(image, gray_image, COLOR_RGB2GRAY);
	else
		gray_image = image.clone();

	//ת����0-1֮��ĸ����������ݣ�����������Ĵ���
	Mat floatImage;
	//float_image=(float)gray_image*(1.0/255.0)
	gray_image.convertTo(floatImage, CV_32FC1, 1.0 / 255.0, 0);
	double sig_diff=0;
	if (double_size){
		Mat temp_image;
		resize(floatImage, temp_image, Size(2 * floatImage.cols, 2* floatImage.rows), 0, 0, INTER_LINEAR);
		sig_diff = sqrt(sigma*sigma - 4.0 * INIT_SIGMA*INIT_SIGMA);
		//��˹�˲����ڴ�Сѡ�����Ҫ������ѡ��(4*sig_diff_1+1)-(6*sig_diff+1)֮��
		int kernel_width = 2 * cvRound(GAUSS_KERNEL_RATIO * sig_diff) + 1;
		Size kernel_size(kernel_width, kernel_width);
		GaussianBlur(temp_image, init_image, kernel_size, sig_diff, sig_diff);
	}
	else{
		sig_diff = sqrt(sigma*sigma - 1.0*INIT_SIGMA*INIT_SIGMA);
		//��˹�˲����ڴ�Сѡ�����Ҫ������ѡ��(4*sig_diff_1+1)-(6*sig_diff+1)֮��
		int kernel_width = 2 * cvRound(GAUSS_KERNEL_RATIO * sig_diff) + 1;
		Size kernel_size(kernel_width, kernel_width);
		GaussianBlur(floatImage, init_image, kernel_size, sig_diff, sig_diff);
	}
}

/**************************���ɸ�˹��������һ�ַ���******************************************/
/*init_image��ʾ�Ѿ����ɵĸ�˹��������һ��ͼ��
 nOctaves��ʾ��˹������������
 */
/*void MySift::build_gaussian_pyramid(const Mat &init_image, vector<vector<Mat>> &gauss_pyramid, int nOctaves) const
{
	vector<double> sig;
	sig.push_back(sigma);
	double k = pow(2., 1. / nOctaveLayers);
	for (int i = 1; i < nOctaveLayers+3; ++i){
		double prev_sig = pow(k,(double)(i-1))*sigma;
		double curr_sig = k*prev_sig;
		sig.push_back(sqrt(curr_sig*curr_sig - prev_sig*prev_sig));
	}

	vector<Mat> temp_gauss;
	for (int i = 0; i < nOctaves; ++i)//����ÿһ��
	{
		for (int j = 0; j < nOctaveLayers + 3; ++j)//�������ڵ�ÿһ��
		{
			if (i == 0 && j == 0)//��һ�飬��һ��
				temp_gauss.push_back(init_image);
			else if (j == 0)
			{
				gauss_pyramid.push_back(temp_gauss);//����֮ǰһ��
				temp_gauss.clear();//���֮ǰ��
				Mat down_prev;
				resize(gauss_pyramid[i - 1][3], down_prev, 
					Size(gauss_pyramid[i - 1][3].cols / 2, 
					gauss_pyramid[i - 1][3].rows / 2), 0, 0, INTER_LINEAR);
				temp_gauss.push_back(down_prev);
			}
			else
			{
				Mat curr_gauss;
				GaussianBlur(temp_gauss[j - 1], curr_gauss, Size(), sig[j], sig[j], BORDER_DEFAULT);
				temp_gauss.push_back(curr_gauss);
				if (i == nOctaves - 1 && j == nOctaveLayers + 2)
					gauss_pyramid.push_back(temp_gauss);					
			}
		}
	}
}*/

/**************************���ɸ�˹�������ڶ��ַ���******************************************/
/*init_image��ʾ�Ѿ����ɵĸ�˹��������һ��ͼ��
 gauss_pyramid��ʾ���ɵĸ�˹������
 nOctaves��ʾ��˹������������
*/
void MySift::build_gaussian_pyramid(const Mat &init_image, vector<vector<Mat>> &gauss_pyramid, int nOctaves) const
{
	vector<double> sig;
	sig.push_back(sigma);
	double k = pow(2.0, 1.0 / nOctaveLayers);
	for (int i = 1; i < nOctaveLayers + 3; ++i){
		double prev_sig = pow(k,double(i-1))*sigma;
		double curr_sig = k*prev_sig;
		sig.push_back(sqrt(curr_sig*curr_sig - prev_sig*prev_sig));
	}

	gauss_pyramid.resize(nOctaves);
	for (int i = 0; i < nOctaves; ++i)
	{
		gauss_pyramid[i].resize(nOctaveLayers + 3);
	}

	for (int i = 0; i < nOctaves; ++i)//����ÿһ��
	{
		for (int j = 0; j < nOctaveLayers + 3; ++j)//�������ڵ�ÿһ��
		{
			if (i == 0 && j == 0)//��һ�飬��һ��
				gauss_pyramid[0][0] = init_image;
			else if (j == 0)
			{
				resize(gauss_pyramid[i - 1][3], gauss_pyramid[i][0],
					Size(gauss_pyramid[i - 1][3].cols / 2,
					gauss_pyramid[i - 1][3].rows / 2), 0, 0, INTER_LINEAR);
			}
			else
			{
				//��˹�˲����ڴ�Сѡ�����Ҫ������ѡ��(4*sig_diff_1+1)-(6*sig_diff+1)֮��
				int kernel_width = 2 * cvRound(GAUSS_KERNEL_RATIO * sig[j]) + 1;
				Size kernel_size(kernel_width, kernel_width);
				GaussianBlur(gauss_pyramid[i][j - 1], gauss_pyramid[i][j], kernel_size, sig[j], sig[j]);
			}
		}
	}
}

/*******************���ɸ�˹��ֽ���������LOG������*************************/
/*dog_pyramid��ʾDOG������
 gauss_pyramin��ʾ��˹������*/
void MySift::build_dog_pyramid(vector<vector<Mat>> &dog_pyramid, const vector<vector<Mat>> &gauss_pyramid) const
{
	vector<vector<Mat>>::size_type nOctaves = gauss_pyramid.size();
	for (vector<vector<Mat>>::size_type i = 0; i < nOctaves; ++i)
	{
		vector<Mat> temp_vec;
		for (auto j = 0; j < nOctaveLayers + 2; ++j)
		{
			Mat temp_img = gauss_pyramid[i][j + 1] - gauss_pyramid[i][j];
			temp_vec.push_back(temp_img);
		}
		dog_pyramid.push_back(temp_vec);
		temp_vec.clear();
	}
}


/***********************�ú�������߶ȿռ��������������***************************/
/*image��ʾ����������λ�õĸ�˹ͼ��
 pt��ʾ�������λ������(x,y)
 scale������ĳ߶�
 n��ʾֱ��ͼbin����
 hist��ʾ����õ���ֱ��ͼ
 ��������ֵ��ֱ��ͼhist�е������ֵ*/
static float clac_orientation_hist(const Mat &image, Point pt, float scale, int n, float *hist)
{
	int radius = cvRound(ORI_RADIUS*scale);//����������뾶(3*1.5*scale)
	int len = (2 * radius + 1)*(2 * radius + 1);//���������������ܸ��������ֵ��

	float sigma = ORI_SIG_FCTR*scale;//�����������˹Ȩ�ر�׼��(1.5*scale)
	float exp_scale = -1.f / (2 * sigma*sigma);

	//ʹ��AutoBuffer����һ���ڴ棬������4���ռ��Ŀ����Ϊ�˷������ƽ��ֱ��ͼ����Ҫ
	AutoBuffer<float> buffer(4 * len + n + 4);
	//X����ˮƽ��֣�Y������ֵ��֣�Mag�����ݶȷ��ȣ�Ori�����ݶȽǶȣ�W�����˹Ȩ��
	float *X = buffer, *Y = buffer + len, *Mag = Y, *Ori = Y + len, *W = Ori + len;
	float *temp_hist = W + len + 2;//��ʱ����ֱ��ͼ����

	for (int i = 0; i < n; ++i)
		temp_hist[i] = 0.f;//��������

	//�����������ص�ˮƽ��ֺ���ֱ���
	int k = 0;
	for (int i = -radius; i < radius; ++i)
	{
		int y = pt.y + i;
		if (y<=0 || y>=image.rows - 1)
			continue;
		for (int j = -radius; j < radius; ++j)
		{
			int x = pt.x + j;
			if (x<=0 || x>=image.cols - 1)
				continue;

			float dx = image.at<float>(y,x+1) - image.at<float>(y,x-1);
			float dy = image.at<float>(y + 1, x) - image.at<float>(y - 1, x);
			X[k] = dx; Y[k] = dy; W[k] = (i*i + j*j)*exp_scale;
			++k;
		}
	}

	len = k;
	//�����������ص��ݶȷ���,�ݶȷ��򣬸�˹Ȩ��
	//exp(W, W, len);
	//fastAtan2(Y, X, Ori, len, true);//�Ƕȷ�Χ0-360��
	//magnitude(X, Y, Mag, len);

	for (int k = 0; k < len; k++)
	{
		Ori[k] = fastAtan2(Y[k], X[k]);
		Mag[k] = sqrt(X[k] * X[k] + Y[k] * Y[k]);
		W[k] = exp(W[k]);
	}

	for (int i = 0; i < len; ++i)
	{
		int bin = cvRound((n / 360.f)*Ori[i]);//bin�ķ�ΧԼ����[0,(n-1)]
		if (bin >= n)
			bin = bin - n;
		if (bin < 0)
			bin = bin + n;
		temp_hist[bin] = temp_hist[bin] + Mag[i] * W[i];
	}
	
	//ƽ��ֱ��ͼ
	temp_hist[-1] = temp_hist[n - 1];
	temp_hist[-2] = temp_hist[n - 2];
	temp_hist[n] = temp_hist[0];
	temp_hist[n + 1] = temp_hist[1];
	for (int i = 0; i < n; ++i)
	{
		hist[i] = (temp_hist[i - 2] + temp_hist[i + 2])*(1.f / 16.f) +
			(temp_hist[i - 1] + temp_hist[i + 1])*(4.f / 16.f) +
			temp_hist[i] * (6.f / 16.f);
	}

	//���ֱ��ͼ�����ֵ
	float max_value = hist[0];
	for (int i = 1; i < n; ++i)
	{
		if (hist[i]>max_value)
			max_value = hist[i];
	}
	return max_value;
}

/****************************�ú�����ȷ��λ������λ��(x,y,scale)*************************/
/*dog_pry��ʾDOG������
 kpt��ʾ��ȷ��λ������������Ϣ
 octave��ʾ��ʼ���������ڵ���
 layer��ʾ��ʼ���������ڵĲ�
 row��ʾ��ʼ��������ͼ���е�������
 col��ʾ��ʼ��������ͼ���е�������
 nOctaveLayers��ʾDOG������ÿ���м������Ĭ����3
 contrastThreshold��ʾ�Աȶ���ֵ��Ĭ����0.04
 edgeThreshold��ʾ��Ե��ֵ��Ĭ����10
 sigma��ʾ��˹�߶ȿռ���ײ�ͼ��߶ȣ�Ĭ����1.6*/
static bool adjust_local_extrema(const vector<vector<Mat>> &dog_pyr, KeyPoint &kpt, int octave, int &layer,
	int &row, int &col, int nOctaveLayers, float contrastThreshold, float edgeThreshold, float sigma)
{
	float xi = 0, xr = 0, xc = 0;
	int i = 0;
	for ( ; i < MAX_INTERP_STEPS; ++i)//����������
	{
		const Mat &img = dog_pyr[octave][layer];//��ǰ��ͼ������
		const Mat &prev = dog_pyr[octave][layer - 1];//֮ǰ��ͼ������
		const Mat &next = dog_pyr[octave][layer + 1];//��һ��ͼ������

		//������λ��x����y����,�߶ȷ����һ��ƫ����
		float dx = (img.at<float>(row, col + 1) - img.at<float>(row, col - 1))*(1.f / 2.f);
		float dy = (img.at<float>(row + 1, col) - img.at<float>(row - 1, col))*(1.f / 2.f);
		float dz = (next.at<float>(row, col) - prev.at<float>(row, col))*(1.f / 2.f);

		//����������λ�ö���ƫ����
		float v2 = img.at<float>(row, col);
		float dxx = img.at<float>(row, col + 1) + img.at<float>(row, col - 1) - 2 * v2;
		float dyy = img.at<float>(row + 1, col) + img.at<float>(row - 1, col) - 2 * v2;
		float dzz = prev.at<float>(row, col) + next.at<float>(row, col) - 2 * v2;

		//������������Χ��϶���ƫ����
		float dxy = (img.at<float>(row + 1, col + 1) + img.at<float>(row - 1, col - 1) -
			img.at<float>(row + 1, col - 1) - img.at<float>(row - 1, col + 1))*(1.f / 4.f);
		float dxz = (next.at<float>(row, col + 1) + prev.at<float>(row, col - 1) -
			next.at<float>(row, col - 1) - prev.at<float>(row, col +1))*(1.f / 4.f);
		float dyz = (next.at<float>(row+1, col) + prev.at<float>(row-1, col) -
			next.at<float>(row-1, col) - prev.at<float>(row+1, col))*(1.f / 4.f);

		Matx33f H (dxx, dxy, dxz, 
			       dxy, dyy, dyz, 
				dxz, dyz, dzz);

		Vec3f dD(dx, dy, dz);

		Vec3f X = H.solve(dD, DECOMP_SVD);
		
		xc = -X[0];//x����ƫ����
		xr = -X[1];//y����ƫ����
		xi = -X[2];//�߶ȷ���ƫ����

		//�����������ƫ������С��0.5��˵���Ѿ��ҵ�������׼ȷλ��
		if (abs(xc) < 0.5f && abs(xr) < 0.5f && abs(xi) < 0.5f)
			break;

		//�������һ�������ƫ����������ɾ���õ�
		if (abs(xc)>(float)(INT_MAX / 3) ||
			abs(xr)>(float)(INT_MAX / 3) ||
			abs(xi)>(float)(INT_MAX / 3))
			return false;

		col = col + cvRound(xc);
		row = row + cvRound(xr);
		layer = layer + cvRound(xi);

		//��������㶨λ�ڱ߽�����ͬ��Ҳ��Ҫɾ��
		if (layer<1 || layer>nOctaveLayers ||
			col<IMG_BORDER || col>img.cols - IMG_BORDER ||
			row<IMG_BORDER || row>img.rows - IMG_BORDER)
			return false;
	}

	//���i=MAX_INTERP_STEPS��˵��ѭ������Ҳû������������������������Ҫ��ɾ��
	if (i >= MAX_INTERP_STEPS)
		return false;

	/**************************�ٴ�ɾ������Ӧ��********************************/
	//�ٴμ���������λ��x����y����,�߶ȷ����һ��ƫ����
	{
		const Mat &img = dog_pyr[octave][layer];
		const Mat &prev = dog_pyr[octave][layer - 1];
		const Mat &next = dog_pyr[octave][layer + 1];

		float dx = (img.at<float>(row, col + 1) - img.at<float>(row, col - 1))*(1.f / 2.f);
		float dy = (img.at<float>(row + 1, col) - img.at<float>(row - 1, col))*(1.f / 2.f);
		float dz = (next.at<float>(row, col) - prev.at<float>(row, col))*(1.f / 2.f);
		Matx31f dD(dx, dy, dz);
		float t = dD.dot(Matx31f(xc, xr, xi));

		float contr = img.at<float>(row, col) + t*0.5f;//��������Ӧ
		//Low����contr��ֵ��0.03������RobHess�Ƚ�����ֵΪ0.04/nOctaveLayers
		if (abs(contr) < contrastThreshold / nOctaveLayers)
			return false;


		/******************************ɾ����Ե��Ӧ�Ƚ�ǿ�ĵ�************************************/
		//�ٴμ���������λ�ö���ƫ����
		float v2 = img.at<float>(row, col);
		float dxx = img.at<float>(row, col + 1) + img.at<float>(row, col - 1) - 2 * v2;
		float dyy = img.at<float>(row + 1, col) + img.at<float>(row - 1, col) - 2 * v2;
		float dxy = (img.at<float>(row + 1, col + 1) + img.at<float>(row - 1, col - 1) -
			img.at<float>(row + 1, col - 1) - img.at<float>(row - 1, col + 1))*(1.f / 4.f);
		float det = dxx*dyy - dxy*dxy;
		float trace = dxx + dyy;
		if (det < 0 || (trace*trace*edgeThreshold >= det*(edgeThreshold + 1)*(edgeThreshold + 1)))
			return false;

		/*********��ĿǰΪֹ��������������������Ҫ�󣬱������������Ϣ***********/
		kpt.pt.x = ((float)col + xc)*(1<<octave);//�������ײ��ͼ���x����
		kpt.pt.y = ((float)row + xr)*(1<<octave);//�������ײ�ͼ���y����
		kpt.octave = octave + (layer << 8);//��ű����ڵ��ֽڣ���ű����ڸ��ֽ�
		//�������ײ�ͼ��ĳ߶�
		kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1<<octave);
		kpt.response = abs(contr);//��������Ӧֵ

		return true;
	}

}


/************�ú�����DOG�������Ͻ����������⣬�����㾫ȷ��λ��ɾ���ͶԱȶȵ㣬ɾ����Ե��Ӧ�ϴ��**********/
/*dog_pyr��ʾ��˹������
 gauss_pyr��ʾ��˹������
 keypoints��ʾ��⵽��������*/
void MySift::find_scale_space_extrema(const vector<vector<Mat>> &dog_pyr, const vector<vector<Mat>> &gauss_pyr,
	vector<KeyPoint> &keypoints) const
{
	int nOctaves = (int)dog_pyr.size();
	//Low���½���threshold��0.03��Rob Hess����ʹ��0.04/nOctaveLayers��Ϊ��ֵ
	float threshold = (float)(contrastThreshold / nOctaveLayers);
	const int n = ORI_HIST_BINS;//n=36
	float hist[n];
	KeyPoint kpt;

	keypoints.clear();//�����keypoints
	//int numKeys = 0;

	for (int i = 0; i < nOctaves; ++i)//����ÿһ��
	{
		for (int j = 1; j <= nOctaveLayers; ++j)//��������ÿһ��
		{
			const Mat &curr_img = dog_pyr[i][j];//��ǰ��
			const Mat &prev_img = dog_pyr[i][j - 1];//֮ǰ��
			const Mat &next_img = dog_pyr[i][j + 1];
			int num_row = curr_img.rows;
			int num_col = curr_img.cols;//��õ�ǰ��ͼ��Ĵ�С
			size_t step = curr_img.step1();//һ��Ԫ����ռ���

			for (int r = IMG_BORDER; r < num_row - IMG_BORDER; ++r)
			{
				const float *curr_ptr = curr_img.ptr<float>(r);
				const float *prev_ptr = prev_img.ptr<float>(r);
				const float *next_ptr = next_img.ptr<float>(r);

				for (int c = IMG_BORDER; c < num_col - IMG_BORDER; ++c)
				{
					float val = curr_ptr[c];//��ǰ���ĵ���Ӧֵ

					//��ʼ���������
					if (abs(val)>threshold &&
						((val > 0 && val >= curr_ptr[c - 1] && val >= curr_ptr[c + 1] &&
						val >= curr_ptr[c - step - 1] && val >= curr_ptr[c - step] && val >= curr_ptr[c - step + 1] &&
						val >= curr_ptr[c + step - 1] && val >= curr_ptr[c + step] && val >= curr_ptr[c + step + 1] &&
						val >= prev_ptr[c] && val >= prev_ptr[c - 1] && val >= prev_ptr[c + 1] &&
						val >= prev_ptr[c - step - 1] && val >= prev_ptr[c - step] && val >= prev_ptr[c - step + 1] &&
						val >= prev_ptr[c + step - 1] && val >= prev_ptr[c + step] && val >= prev_ptr[c + step + 1] &&
						val >= next_ptr[c] && val >= next_ptr[c - 1] && val >= next_ptr[c + 1] &&
						val >= next_ptr[c - step - 1] && val >= next_ptr[c - step] && val >= next_ptr[c - step + 1] &&
						val >= next_ptr[c + step - 1] && val >= next_ptr[c + step] && val >= next_ptr[c + step + 1])  ||
						(val < 0 && val <= curr_ptr[c - 1] && val <= curr_ptr[c + 1] &&
						val <= curr_ptr[c - step - 1] && val <= curr_ptr[c - step] && val <= curr_ptr[c - step + 1] &&
						val <= curr_ptr[c + step - 1] && val <= curr_ptr[c + step] && val <= curr_ptr[c + step + 1] &&
						val <= prev_ptr[c] && val <= prev_ptr[c - 1] && val <= prev_ptr[c + 1] &&
						val <= prev_ptr[c - step - 1] && val <= prev_ptr[c - step] && val <= prev_ptr[c - step + 1] &&
						val <= prev_ptr[c + step - 1] && val <= prev_ptr[c + step] && val <= prev_ptr[c + step + 1] &&
						val <= next_ptr[c] && val <= next_ptr[c - 1] && val <= next_ptr[c + 1] &&
						val <= next_ptr[c - step - 1] && val <= next_ptr[c - step] && val <= next_ptr[c - step + 1] &&
						val <= next_ptr[c + step - 1] && val <= next_ptr[c + step] && val <= next_ptr[c + step + 1])))
					{
						//++numKeys;
						//����������ʼ�кţ��кţ���ţ����ڲ��
						int r1 = r, c1 = c, octave = i, layer = j;
						if (!adjust_local_extrema(dog_pyr, kpt, octave, layer, r1, c1,
							nOctaveLayers, (float)contrastThreshold, 
							(float)edgeThreshold, (float)sigma))
						{
							continue;//����ó�ʼ�㲻�����������򲻱���ĵ�
						}

						float scale = kpt.size / float (1 << octave);//������������ڱ���ĳ߶�
						float max_hist = clac_orientation_hist(gauss_pyr[octave][layer], 
							Point(c1, r1), scale, n, hist);
						float mag_thr = max_hist*ORI_PEAK_RATIO;

						for (int i = 0; i < n; ++i)
						{
							int left=0, right=0;
							if (i == 0)
								left = n - 1;
							else
								left = i - 1;

							if (i == n - 1)
								right = 0;
							else
								right = i + 1;

							if (hist[i] > hist[left] && hist[i] > hist[right] && hist[i] >= mag_thr)
							{
								float bin = i + 0.5f*(hist[left] - hist[right]) / (hist[left] + hist[right] - 2 * hist[i]);
								if (bin < 0)
									bin = bin + n;
								if (bin >= n)
									bin = bin - n;

								kpt.angle = (360.f / n)*bin;//�������������0-360��
								keypoints.push_back(kpt);//�����������
								
							}

						}
					}
				}
			}
		}
	}

	//cout << "��ʼ����Ҫ�������������: " << numKeys << endl;
}


/******************************����һ���������������***********************************/
/*gauss_image��ʾ���������ڵĸ�˹ͼ��
 main_angle��ʾ������������򣬽Ƕȷ�Χ��0-360��
 pt��ʾ�������ڸ�˹ͼ���ϵ����꣬����뱾�飬�����������ײ�
 scale��ʾ���������ڲ�ĳ߶ȣ�����ڱ��飬�����������ײ�
 d��ʾ����������������
 n��ʾÿ�������������ݶȽǶȵȷָ���
 descriptor��ʾ���ɵ��������������*/
static void calc_sift_descriptor(const Mat &gauss_image, float main_angle, Point2f pt,
	float scale, int d, int n, float *descriptor)
{
	Point ptxy(cvRound(pt.x), cvRound(pt.y));//����ȡ��
	float cos_t = cosf(-main_angle*(float)(CV_PI / 180));
	float sin_t = sinf(-main_angle*(float)(CV_PI / 180));
	float bins_per_rad = n / 360.f;//n=8
	float exp_scale = -1.f / (d*d*0.5f);
	float hist_width = DESCR_SCL_FCTR*scale;//ÿ������Ŀ��
	int radius = cvRound(hist_width*(d + 1)*sqrt(2)*0.5f);//����������뾶

	int rows = gauss_image.rows, cols = gauss_image.cols;
	radius = min(radius, (int)sqrt((double)rows*rows + cols*cols));
	cos_t = cos_t / hist_width;
	sin_t = sin_t / hist_width;

	int len = (2 * radius + 1)*(2 * radius + 1);
	int histlen = (d + 2)*(d + 2)*(n + 2);
	
	AutoBuffer<float> buf(6 * len + histlen);
	//X����ˮƽ��֣�Y������ֱ��֣�Mag�����ݶȷ��ȣ�Angle���������㷽��,W�����˹Ȩ��
	float *X = buf, *Y = buf + len, *Mag = Y, *Angle = Y + len, *W = Angle + len;
	float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;

	//�������ֱ��ͼ����
	for (int i = 0; i < d + 2; ++i)
	{
		for (int j = 0; j < d + 2; ++j)
		{
			for (int k = 0; k < n+2; ++k)
				hist[(i*(d + 2) + j)*(n + 2) + k] = 0.f;
		}	
	}

	//��������������Χ��ÿ�����صĲ�ֺ˸�˹Ȩ�ص�ָ������
	int k = 0;
	for (int i = -radius; i < radius; ++i)
	{
		for (int j = -radius; j < radius; ++j)
		{
			float c_rot = j*cos_t - i*sin_t;
			float r_rot = j*sin_t + i*cos_t;
			float rbin = r_rot + d / 2 - 0.5f;
			float cbin = c_rot + d / 2 - 0.5f;
			int r = ptxy.y + i, c = ptxy.x + j;

			//����rbin,cbin��Χ��(-1,d)
			if (rbin>-1 && rbin<d && cbin>-1 && cbin<d &&
				r>0 && r<rows - 1 && c>0 && c < cols - 1)
			{
				float dx = gauss_image.at<float>(r, c + 1) - gauss_image.at<float>(r, c - 1);
				float dy = gauss_image.at<float>(r + 1, c) - gauss_image.at<float>(r - 1, c);
				X[k] = dx; //ˮƽ���
				Y[k] = dy;//��ֱ���
				RBin[k] = rbin;
				CBin[k]=cbin;
				W[k] = (c_rot*c_rot + r_rot*r_rot)*exp_scale;//��˹Ȩֵ��ָ������
				++k;
			}
		}
	}

	//���������ݶȷ��ȣ��ݶȽǶȣ��͸�˹Ȩֵ
	len = k;
	//fastAtan2(Y, X, Angle, len, true);//�Ƕȷ�Χ��0-360��
	//magnitude(X, Y, Mag, len);//����
	for (int k = 0; k < len; k++)
	{
		Angle[k] = fastAtan2(Y[k], X[k]);
		Mag[k] = sqrt(X[k] * X[k] + Y[k] * Y[k]);
		W[k] = exp(W[k]);
	}


	//exp(W, W, len);//��˹Ȩֵ

	//����ÿ���������������
	for (k = 0; k < len; ++k)
	{
		float rbin = RBin[k], cbin = CBin[k];//rbin,cbin��Χ��(-1,d)
		float obin = (Angle[k] - main_angle)*bins_per_rad;
		float mag = Mag[k] * W[k];

		int r0 = cvFloor(rbin);//roȡֵ������{-1,0,1,2��3}
		int c0 = cvFloor(cbin);//c0ȡֵ������{-1��0��1��2��3}
		int o0 = cvFloor(obin);
		rbin = rbin - r0;
		cbin = cbin - c0;
		obin = obin - o0;

		//���Ʒ�ΧΪ[0,n)
		if (o0 < 0)
			o0 = o0 + n;
		if (o0 >= n)
			o0 = o0 - n;

		//ʹ�������Բ�ֵ����������ֱ��ͼ
		float v_r1 = mag*rbin;//�ڶ��з����ֵ
		float v_r0 = mag - v_r1;//��һ�з����ֵ

		float v_rc11 = v_r1*cbin;
		float v_rc10 = v_r1 - v_rc11;
		float v_rc01 = v_r0*cbin;
		float v_rc00 = v_r0 - v_rc01;

		float v_rco111 = v_rc11*obin;
		float v_rco110 = v_rc11 - v_rco111;

		float v_rco101 = v_rc10*obin;
		float v_rco100 = v_rc10 - v_rco101;

		float v_rco011 = v_rc01*obin;
		float v_rco010 = v_rc01 - v_rco011;

		float v_rco001 = v_rc00*obin;
		float v_rco000 = v_rc00 - v_rco001;

		//�������������������
		int idx = ((r0 + 1)*(d + 2) + c0 + 1)*(n + 2) + o0;
		hist[idx] += v_rco000;
		hist[idx + 1] += v_rco001;
		hist[idx + n + 2] += v_rco010;
		hist[idx + n + 3] += v_rco011;
		hist[idx + (d + 2)*(n + 2)] += v_rco100;
		hist[idx + (d + 2)*(n + 2) + 1] += v_rco101;
		hist[idx + (d + 3)*(n + 2)] += v_rco110;
		hist[idx + (d + 3)*(n + 2) + 1] += v_rco111;
	}

	//����Բ��ѭ�������ԣ��Լ����Ժ����С�� 0 �Ȼ���� 360 �ȵ�ֵ���½��е�����ʹ
	//���� 0��360 ��֮��
	for (int i = 0; i < d; ++i)
	{
		for (int j = 0; j < d; ++j)
		{
			int idx = ((i + 1)*(d + 2) + (j + 1))*(n + 2);
			hist[idx] += hist[idx + n];
			//hist[idx + 1] += hist[idx + n + 1];//opencvԴ������仰�Ƕ����,hist[idx + n + 1]��Զ��0.0
			for (k = 0; k < n; ++k)
				descriptor[(i*d + j)*n + k] = hist[idx + k];
		}
	}

	//�������ӽ��й�һ��
	int lenght = d*d*n;
	float norm = 0;
	for (int i = 0; i < lenght; ++i)
	{
		norm = norm + descriptor[i] * descriptor[i];
	}
	norm = sqrt(norm);
	for (int i = 0; i < lenght; ++i)
	{
		descriptor[i] = descriptor[i] / norm;
	}

	//��ֵ�ض�
	for (int i = 0; i < lenght; ++i)
	{
		descriptor[i] = min(descriptor[i], DESCR_MAG_THR);
	}

	//�ٴι�һ��
	norm = 0;
	for (int i = 0; i < lenght; ++i)
	{
		norm = norm + descriptor[i] * descriptor[i];
	}
	norm = sqrt(norm);
	for (int i = 0; i < lenght; ++i)
	{
		descriptor[i] = descriptor[i] /norm;
	}	

}


/********************************�ú���������������������������***************************/
/*gauss_pyr��ʾ��˹������
 keypoints��ʾ�����㡢
 descriptors��ʾ���ɵ��������������*/
void MySift::calc_descriptors(const vector<vector<Mat>> &gauss_pyr, vector<KeyPoint> &keypoints,
	Mat &descriptors) const
{
	int d = DESCR_WIDTH;//d=4,�������������������d x d
	int n = DESCR_HIST_BINS;//n=8,ÿ�������������ݶȽǶȵȷ�Ϊ8������
	descriptors.create(keypoints.size(), d*d*n, CV_32FC1);//����ռ�

	for (size_t i = 0; i < keypoints.size(); ++i)//����ÿһ��������
	{
		int octaves, layer;
		//�õ����������ڵ���ţ����
		octaves = keypoints[i].octave & 255;
		layer = (keypoints[i].octave >> 8) & 255;

		//�õ�����������ڱ�������꣬������ײ�
		Point2f pt(keypoints[i].pt.x/(1<<octaves), keypoints[i].pt.y/(1<<octaves));
		float scale = keypoints[i].size / (1 << octaves);//�õ�����������ڱ���ĳ߶�
		float main_angle = keypoints[i].angle;//������������

		//����ĵ��������
		calc_sift_descriptor(gauss_pyr[octaves][layer],
			main_angle, pt, scale,
			d, n, descriptors.ptr<float>((int)i));

		if (double_size)//���ͼ��ߴ�����һ��
		{
			keypoints[i].pt.x = keypoints[i].pt.x / 2.f;
			keypoints[i].pt.y = keypoints[i].pt.y / 2.f;
		}
	}
		
}

/******************************��������*********************************/
/*image��ʾ�����ͼ��
 gauss_pyr��ʾ���ɵĸ�˹������
 dog_pyr��ʾ���ɵĸ�˹���DOG������
 keypoints��ʾ��⵽��������*/
void MySift::detect(const Mat &image, vector<vector<Mat>> &gauss_pyr, vector<vector<Mat>> &dog_pyr,
	 vector<KeyPoint> &keypoints) const
{
	if (image.empty() || image.depth() != CV_8U)
		CV_Error(CV_StsBadArg,"����ͼ��Ϊ�գ�����ͼ����Ȳ���CV_8U");

	
	//�����˹����������
	int nOctaves;
	nOctaves = num_octaves(image);

	//���ɸ�˹��������һ��ͼ��
	Mat init_gauss;
	create_initial_image(image, init_gauss);

	//���ɸ�˹�߶ȿռ�ͼ��
	build_gaussian_pyramid(init_gauss, gauss_pyr, nOctaves);

	//���ɸ�˹��ֽ�����(DOG��������or LOG������)
	build_dog_pyramid(dog_pyr, gauss_pyr);

	//��DOG�������ϼ��������
	find_scale_space_extrema(dog_pyr, gauss_pyr, keypoints);

	//���ָ�������������,�����趨������С��Ĭ�ϼ������������
	if (nfeatures!=0 && nfeatures < (int)keypoints.size())
	{
		//��������Ӧֵ�Ӵ�С����
		sort(keypoints.begin(), keypoints.end(),
			[](const KeyPoint &a, const KeyPoint &b)
		{return a.response>b.response; });

		//ɾ��������������
		keypoints.erase(keypoints.begin()+nfeatures,keypoints.end());
	}


}

/**********************����������*******************/
/*gauss_pyr��ʾ��˹������
 keypoints��ʾ�����㼯��
 descriptors��ʾ�������������*/
void MySift::comput_des(const vector<vector<Mat>> &gauss_pyr, vector<KeyPoint> &keypoints,Mat &descriptors) const
{
	calc_descriptors(gauss_pyr, keypoints, descriptors);
}

