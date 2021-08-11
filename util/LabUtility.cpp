#include<bits/stdc++.h>
#include<random>
using namespace std;

#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace cv;

random_device rd;
mt19937 mt(rd());
uniform_int_distribution<int> d(0, 1e9);

typedef long long ll;
char fileName[1010];

string rootDir = "./20210803/";
string predDir = rootDir + "res1/";
string resDir = rootDir + "concat/";

const double PI = 3.141592653589793;
const double EXP = 2.71828182845904;
const double SIGMA = 1;
const double SIGMA_COEF = 12.0;
const double COLOR_SIGMA = 64;
const int HEIGHT = 256;
const int WIDTH = 256;
vector<double> normSrc[256][256];
vector<double> normTarget[256][256];
double GaussianFunction(double x, double sigma = SIGMA) {
	const double tmp = 1.0 / ((sqrt(2 * PI)) * sigma);
	return tmp * exp(-(x * x) / (2 * sigma * sigma));
}
double GaussianFunction2D(double x, double y, double sigma = SIGMA) {
	return GaussianFunction(x, sigma) * GaussianFunction(y, sigma);
}
void BackgroundCorrection(Mat& gt, Mat& pred, int flag) {
	/*
	0: Magenta: (255, 0, 255)
	1: Gray: (127, 127, 127)
	*/
	vector<vector<int>> back = { {255, 0, 255}, {127, 127, 127} };
	auto it1 = gt.begin<Vec4b>();
	auto it2 = pred.begin<Vec4b>();
	for (; it1 != gt.end<Vec4b>(); ++it1, ++it2) {
		bool f = 1;
		for (int i = 0; i < 3; ++i) {
			if ((*it1)[i] != back[0][i]) {
				f = 0;
				break;
			}
		}
		if (f)
			for (int i = 0; i < 3; ++i)
				(*it2)[i] = back[flag][i];
	}
}
vector<vector<double>> GetGaussianFilter(int kernelSize, double sigma = -1) {
	if (sigma == -1)
		sigma = kernelSize / SIGMA_COEF;

	vector<vector<double>> kernel(kernelSize);
	for (int i = 0; i < kernelSize; ++i) {
		for (int j = 0; j < kernelSize; ++j) {
			kernel[i].push_back(GaussianFunction2D(i - kernelSize / 2, j - kernelSize / 2, sigma));
		}
	}
	return kernel;
}
Mat MyGaussianFiltering(Mat src, int kernelSize = 5) {
	auto filter = GetGaussianFilter(kernelSize, kernelSize / SIGMA_COEF);
	Mat res = Mat(src.rows, src.cols, CV_8UC3);
	for (int i = 0; i < res.rows; ++i) {
		for (int j = 0; j < res.cols; ++j) {
			for (int c = 0; c < 3; ++c) {
				double sum = 0.0;
				int cnt = 0;
				for (int k = 0; k < kernelSize; ++k) {
					for (int l = 0; l < kernelSize; ++l) {
						int nx = i + (k - kernelSize / 2);
						int ny = j + (l - kernelSize / 2);
						if (0 <= nx && nx < res.rows && 0 <= ny && ny < res.cols) {
							sum += filter[k][l] * src.at<Vec3b>(nx, ny)[c];
						}
					}
				}
				res.at<Vec3b>(i, j)[c] = sum;
			}

		}
	}
	return res;
}
vector<vector<Vec3f>> MatNormalization(Mat img) {
	vector<vector<Vec3f>> res(img.rows);
	for (int i = 0; i < img.rows; ++i) {
		res[i].resize(img.cols);
		for (int j = 0; j < img.cols; ++j) {
			auto cur = img.at<Vec3b>(i, j);
			double d = sqrt(cur[0] * cur[0] + cur[1] * cur[1] + cur[2] * cur[2]);
			for (int k = 0; k < 3; ++k) {
				res[i][j][k] = cur[k] / d;
			}
		}
	}
	return res;
}
Mat JointBilateralFiltering(Mat &src, Mat &target, int kernelSize = 23, double sigma = -1, double colorSigma = -1) {
	if (sigma == -1)
		sigma = kernelSize / SIGMA_COEF;
	if (colorSigma == -1)
		colorSigma = COLOR_SIGMA;
	auto filter = GetGaussianFilter(kernelSize, sigma);
	Mat res = Mat(src.rows, src.cols, CV_8UC3);
	for (int i = 0; i < res.rows; ++i) {
		for (int j = 0; j < res.cols; ++j) {
			for (int c = 0; c < 3; ++c) {
				double sum = 0.0;
				double sum2 = 0.0;
				int cnt = 0;
				for (int k = 0; k < kernelSize; ++k) {
					for (int l = 0; l < kernelSize; ++l) {
						int nx = i + (k - kernelSize / 2);
						int ny = j + (l - kernelSize / 2);

						auto origin = src.at<Vec3b>(i, j)[c];

						if (0 <= nx && nx < res.rows && 0 <= ny && ny < res.cols) {
							auto np = src.at<Vec3b>(nx, ny)[c];
							sum += filter[k][l] * target.at<Vec3b>(nx, ny)[c] * GaussianFunction(origin - np, colorSigma);
							sum2 += filter[k][l] * GaussianFunction(origin - np, colorSigma);
						}
					}
				}
				res.at<Vec3b>(i, j)[c] = sum / sum2;
			}

		}
	}
	return res;
}
Mat JointBilateralFilteringWithNormalization(Mat src, Mat target, int kernelSize = 9, double sigma = -1, double colorSigma = -1) {
	Mat srcNorm, targetNorm;
	src.convertTo(srcNorm, CV_32FC3, 1.0 / 255);
	target.convertTo(targetNorm, CV_32FC3, 1.0 / 255);
	if (sigma == -1)
		sigma = kernelSize / SIGMA_COEF;
	if (colorSigma == -1)
		colorSigma = 1.0;

	auto filter = GetGaussianFilter(kernelSize, sigma);
	Mat res = Mat(src.rows, src.cols, CV_32FC3);
	for (int i = 0; i < res.rows; ++i) {
		for (int j = 0; j < res.cols; ++j) {
			for (int c = 0; c < 3; ++c) {
				double sum = 0.0;
				double sum2 = 0.0;
				int cnt = 0;
				for (int k = 0; k < kernelSize; ++k) {
					for (int l = 0; l < kernelSize; ++l) {
						int nx = i + (k - kernelSize / 2);
						int ny = j + (l - kernelSize / 2);

						auto origin = srcNorm.at<Vec3f>(i, j)[c];

						if (0 <= nx && nx < res.rows && 0 <= ny && ny < res.cols) {
							auto np = srcNorm.at<Vec3f>(nx, ny)[c];
							sum += filter[k][l] * targetNorm.at<Vec3f>(nx, ny)[c] * GaussianFunction(origin - np, colorSigma);
							sum2 += filter[k][l] * GaussianFunction(origin - np, colorSigma);
						}
					}
				}
				res.at<Vec3f>(i, j)[c] = (sum / sum2) ;
			}

		}
	}
	return res;
}
Mat JointBilateralFilteringWithNormalization2(Mat src, Mat target, int kernelSize = 9, double sigma = -1, double colorSigma = -1) {
	Mat srcNorm, targetNorm;
	//src.convertTo(srcNorm, CV_32FC3, 1.0 / 255);
	//target.convertTo(targetNorm, CV_32FC3, 1.0 / 255);
	if (sigma == -1)
		sigma = kernelSize / SIGMA_COEF;
	if (colorSigma == -1)
		colorSigma = 1.0;

	auto filter = GetGaussianFilter(kernelSize, sigma);

	Mat res = Mat(src.rows, src.cols, CV_8UC3);
	for (int i = 0; i < res.rows; ++i) {
		for (int j = 0; j < res.cols; ++j) {
			normSrc[i][j].clear();
			normTarget[i][j].clear();
			double sum1 = 0.0, sum2 = 0.0;
			for (int c = 0; c < 3; ++c) {
				auto cur1 = src.at<Vec3b>(i, j)[c];
				auto cur2 = target.at<Vec3b>(i, j)[c];
				sum1 = ((double)cur1 * cur1);
				sum2 = ((double)cur2 * cur2);
				normSrc[i][j].push_back(cur1);
				normTarget[i][j].push_back(cur2);
			}
			normSrc[i][j].push_back(sqrt(sum1));
			normTarget[i][j].push_back(sqrt(sum2));
		}
	}

	for (int i = 0; i < res.rows; ++i) {
		for (int j = 0; j < res.cols; ++j) {
			for (int c = 0; c < 3; ++c) {
				double sum = 0.0;
				double sum2 = 0.0;
				for (int k = 0; k < kernelSize; ++k) {
					for (int l = 0; l < kernelSize; ++l) {
						int nx = i + (k - kernelSize / 2);
						int ny = j + (l - kernelSize / 2);

						auto origin = normSrc[i][j][c] / normSrc[i][j].back();

						if (0 <= nx && nx < res.rows && 0 <= ny && ny < res.cols) {
							auto np = normSrc[nx][ny][c] / normTarget[i][j].back();
							sum += filter[k][l] * target.at<Vec3b>(nx, ny)[c] * GaussianFunction(origin - np, colorSigma);
							sum2 += filter[k][l] * GaussianFunction(origin - np, colorSigma);
						}
					}
				}
				res.at<Vec3b>(i, j)[c] = (sum / sum2);
			}

		}
	}
	return res;
}
bool BackgroundCheck(Vec4b &p1, int flag) {
	/*
	Magenta: (255, 0, 255)
	Gray: (127, 127, 127)
	*/
	vector<vector<int>> back = { {255, 0, 255}, {127, 127, 127} };
	for (int i = 0; i < 3; ++i)
		if (p1[i] != back[flag][i]) return 1;
	return 0;
}
double GetRMSE(Mat &img1, Mat &img2, int flag) {
	double res = 0.0;
	int cnt = 0;
	auto it1 = img1.begin<Vec4b>();
	auto it2 = img2.begin<Vec4b>();
	int cc = 0;
	for (; it1 != img1.end<Vec4b>(); ++it1, ++it2) {
		if (!BackgroundCheck((*it1), flag) || !BackgroundCheck((*it2), flag)) continue;
		for (int i = 0; i < 3; ++i) {
			auto p1 = ((*it1)[i] - 127.5) / 127.5;
			auto p2 = ((*it2)[i] - 127.5) / 127.5;
			res += (p1 - p2) * (p1 - p2);
			++cnt;
		}
	}
	return sqrt(res / (cnt * 3.0));
}
Mat pred, origin;
void onChange(int pos, void* param) {
	static int checker = 1;
	
	Mat* pMat = (Mat*)param;
	int l = getTrackbarPos("Threshold(L)", "test");
	int r = getTrackbarPos("Threshold(R)", "test");
	Mat tmp1, tmp2;
	Canny(pred, tmp1, l, r);
	Canny(origin, tmp2, l, r);
	//bitwise_and(tmp1, tmp2, tmp1);
	//GaussianBlur(tmp1, tmp1, Size(11, 11), 1.5);
	imshow("test", tmp1);
	if (checker && r == 255) {
		checker = 0;
		for (auto it = tmp1.begin<Vec3b>(); it != tmp1.end<Vec3b>(); ++it) {
			for (int i = 0; i < 3; ++i) {
				printf("%d ", (*it)[i]);
			}
			puts("");
		}
	}
}
void EdgeTest() {
	pred = imread("pp/6.png", 1);
	origin = imread("pp/4.png", 1);
	imshow("origgin", origin);
	imshow("pred", pred);
	Mat img;
	int lThreshold = 0, rThreshold= 244;
	namedWindow("test");
	createTrackbar("Threshold(L)", "test", &lThreshold, 255, onChange, (void*)& pred);
	createTrackbar("Threshold(R)", "test", &rThreshold, 255, onChange, (void*)& pred);
	waitKey(0);
}
void RMSETest() {
	const int backgroundFlag = 0;
	string rootDir = "./20210811/";
	string predDir = rootDir + "res2/";
	string resDir = rootDir + "concat/";
	int imageCount = 95;
	char fName[1010];
	vector<pair<double, int>> rmses;
	double rmse = 0.0;
	puts("Check RMSE...");
	for (int i = 1; i <= imageCount; ++i) {
		printf("%d/%d\n", i, imageCount);
		Mat inp, gt, pred;
		sprintf(fName, "%s(result%d)Ground_Truth.png", predDir.c_str(), i);
		gt = imread(fName);
		sprintf(fName, "%s(result%d)Input_Image.png", predDir.c_str(), i);
		inp = imread(fName);
		sprintf(fName, "%s(result%d)Predicted_Image.png", predDir.c_str(), i);
		pred = imread(fName);

		pred = JointBilateralFilteringWithNormalization(inp, pred);
		if(pred.type() == CV_32FC3)
			pred.convertTo(pred, CV_8UC3, 255);

		double res = GetRMSE(gt, pred, backgroundFlag);
		rmses.push_back({ res, i });
		rmse += res;
	}
	sort(rmses.begin(), rmses.end());
	rmse /= imageCount;
	sprintf(fName, "%sres.txt", resDir.c_str());
	FILE* fp = fopen(fName, "w");
	int count = 1;
	printf("%lf", rmse);
	fprintf(fp, "total RMSE: %lf\n", rmse);
	puts("Save results...");
	for (auto i : rmses) {
		printf("%d/%d\n", count, imageCount);
		fprintf(fp, "%d: %lf\n", i.second, i.first);
		Mat inp, gt, pred;
		sprintf(fName, "%s(result%d)Ground_Truth.png", predDir.c_str(), i.second);
		gt = imread(fName);
		sprintf(fName, "%s(result%d)Input_Image.png", predDir.c_str(), i.second);
		inp = imread(fName);
		sprintf(fName, "%s(result%d)Predicted_Image.png", predDir.c_str(), i.second);
		pred = imread(fName);

		Mat concatImg = inp;

		auto res = JointBilateralFilteringWithNormalization(inp, pred);
		if (res.type() == CV_32FC3)
			res.convertTo(res, CV_8UC3, 255);

		hconcat(concatImg, gt, concatImg);
		hconcat(concatImg, pred, concatImg);
		hconcat(concatImg, res, concatImg);

		sprintf(fName, "%s%d(%d).png", resDir.c_str(), i.second, count++, i.second);
		imwrite(fName, concatImg);
	}
}
int main() {
	RMSETest(); return 0;
	/*
	auto kern = GetGaussianFilter(255, 32);
	Mat kernImg(Size(255, 255), CV_8UC1);
	for (int i = 0; i < kern.size(); ++i) {
		for (int j = 0; j < kern[i].size(); ++j) {
			kernImg.at<uchar>(i, j) = kern[i][j];
		}
	}
	imshow("kernel", kernImg);
	waitKey(0);
	*/
	//Filtering Test
	/*
	BackgroundCorrection(img1, img1, 1);
	imshow("origin", img1);
	auto res1 = MyGaussianFiltering(img1, 9);
	imshow("My Gaussian", res1);
	Mat res2;
	GaussianBlur(img1, res2, Size(9, 9), 9.0 / SIGMA_COEF);
	imshow("CV Gaussian", res2);
	auto res3 = JointBilateralFiltering(img1, img1, 9, -1, 32);
	imshow("My Bilateral", res3);
	auto res4 = JointBilateralFilteringWithNormalization(img1, img1, 9, -1, 32);
	imshow("My Bilateral-2", res4);
	waitKey(0); return 0;
	*/

	//GetRMSE

	int imageCount = 50;
	char fName[1010];
	double rmse = 0.0;
	vector<pair<double, Mat>> result1, result2;

	int kernelCount = 5;
	int kernelStart = 5;
	vector<double> RMSE(kernelCount + 1);
	int fileCount = 10;

	for (int i = 1; i <= imageCount; ++i) {
		printf("%d/%d\n", i, imageCount);
		Mat inp, gt, pred;
		sprintf(fName, "%s(result%d)Ground_Truth.png", predDir.c_str(), i);
		gt = imread(fName);
		sprintf(fName, "%s(result%d)Input_Image.png", predDir.c_str(), i);
		inp = imread(fName);
		sprintf(fName, "%s(result%d)Predicted_Image.png", predDir.c_str(), i);
		pred = imread(fName);

		Mat concatImg = inp;
		hconcat(concatImg, gt, concatImg);
		hconcat(concatImg, pred, concatImg);
		for (int kk = kernelStart; kk <= kernelStart + 2 * kernelCount; kk += 2) {
			//Mat JBF = JointBilateralFiltering(inp, pred, kk);
			//hconcat(concatImg, JBF, concatImg);
			//auto res2 = GetRMSE(gt, JBF);
			//RMSE[(kk - kernelStart) / 2] += res2;

			Mat JBF = JointBilateralFilteringWithNormalization(inp, pred, kk, (kk - 3) / 6.0);
			Mat tmpImg;
			JBF.convertTo(tmpImg, CV_8UC3, 255);
			hconcat(concatImg, tmpImg, concatImg);
			auto res2 = GetRMSE(gt, tmpImg, 1);
			RMSE[(kk - kernelStart) / 2] += res2;
		}

		auto res1 = GetRMSE(gt, pred, 1);

		result1.push_back({ res1, concatImg });
		rmse += res1;
	}
	::sort(result1.begin(), result1.end(), [](pair<double, Mat> p1, pair<double, Mat> p2) {
		return p1.first < p2.first;
		});
	rmse /= imageCount;
	for(int i = 0; i<RMSE.size(); ++i)
		RMSE[i] /= imageCount;
	printf("%lf\n", rmse);
	for (int i = 0; i < RMSE.size(); ++i) {
		//printf("Kernel size = %d(COEF %lf): %lf\n", kernelStart + i * 2, SIGMA_COEF, RMSE[i]);
		printf("%lf\n", RMSE[i]);
	}
	for (int i = 0; i < result1.size(); ++i) {
		sprintf(fName, "%s%d.png", resDir.c_str(), fileCount++);
		imwrite(fName, result1[i].second);
	}
}
