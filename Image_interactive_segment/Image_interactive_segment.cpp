// Image_interactive_segment.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include "IMAGE.h"
#include <iostream>
#include <opencv2/opencv.hpp>  
#include <string>
#include <vector>


int main()
{
	const char *Image_route = "G:\\Matlab project\\superpixel\\bee_label.jpg";
	cv::Mat img = cv::imread("G:\\Matlab project\\superpixel\\bee.jpg");
	cv::Mat label_img= cv::imread("G:\\Matlab project\\superpixel\\bee_label.jpg");

	const int MAX_WIDTH = 500, MAX_HEIGHT = 400;

	IMAGE image(img);
	std::vector<int> label;
	image.MSAM();

}
