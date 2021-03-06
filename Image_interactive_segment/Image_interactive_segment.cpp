// Image_interactive_segment.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include "IMAGE.h"
#include <iostream>
#include <opencv2/opencv.hpp>  
#include <string>
#include <vector>
#include <fstream>

std::vector<cv::Point> p;
std::vector<int> label;
int color = 0;
int k = 0;
void On_mouse(int event, int x, int y, int flags, void* param)
{
	static bool s_bMouseLButtonDown = false;
	static cv::Point cvCurrPoint = cv::Point(0, 0);

	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
		p.push_back(cv::Point(x, y));
		label.push_back(k);
		cvCurrPoint = cvPoint(x, y);
		cv::circle(*(cv::Mat*)param, cvCurrPoint, 2,CV_RGB((uchar)color>>16,(uchar)color>>8,(uchar)color),-1);
		cv::imshow("Point", *(cv::Mat*)param);
		
		break;

	case  CV_EVENT_LBUTTONUP:
		s_bMouseLButtonDown = false;
		break;

	case CV_EVENT_MOUSEMOVE:
		break;
	}
}
int main()
{
	/*std::string file;
	std::getline(std::cin, file);
	for (int i = 0; i < file.size(); ++i) {
		if ('\\' == file[i]) {
			file[i] = '/';
		}
	}*/
	const char *Image_route = "G:\\Matlab project\\superpixel\\bee_label.jpg";
	cv::Mat img = cv::imread("G:\\Matlab project\\superpixel\\bee.jpg");
	cv::Mat label_img= cv::imread("G:\\Matlab project\\superpixel\\bee_label.jpg");

	IMAGE image(img);
	std::vector<int> label;
	image.MSAM();
	
	/*cv::Mat image(1000, 1000, CV_8UC3, CV_RGB(255, 255, 255));

	cv::namedWindow("Point", CV_WINDOW_AUTOSIZE);
	cv::imshow("Point", image);
	cv::setMouseCallback("Point", On_mouse, &image);
	int c;
	do {
		c = cv::waitKey(0);
		if ((char)c == 'p') { ++k; color += 100; }
	} while (c != 101);

	std::ofstream out("G:\\Matlab project\\Adative K-mean\\data.txt");
	for (int i = 0; i < p.size(); ++i) {
		out << p[i].x << " " << p[i].y <<" "<<label[i]<< std::endl;
	}
	out.close();
	cv::destroyWindow("Point");*/
}
