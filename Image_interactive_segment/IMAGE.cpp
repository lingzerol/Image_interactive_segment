#include "pch.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>
#include <cstdlib>
#include <cstdio>
#include <map>
#include <thread>
#include <condition_variable>

std::vector<std::vector<Pixel>> Object, Background;
std::vector<Pixel> temp;
const char *name = "SAM_label";
int flag = -1;

IMAGE::IMAGE(const cv::Mat &source):done(false){
	rgb_image = source;
	lab_image = rgb_to_lab();
}


double *** IMAGE::rgb_to_lab() {
	double*** lab_image;
	lab_image = new double**[rgb_image.rows];
	for (int i = 0; i < rgb_image.rows; ++i) {
		lab_image[i] = new double*[rgb_image.cols];
		for (int j = 0; j < rgb_image.cols; ++j) {
			double R = rgb_image.ptr(i,j)[0] / 255.0;
			double G = rgb_image.ptr(i,j)[1] / 255.0;
			double B = rgb_image.ptr(i,j)[2] / 255.0;

			if (R > 0.04045) R = pow((R + 0.055) / 1.055, 2.4);
			else R = R / 12.92;
			if (G > 0.04045) G = pow((G + 0.055) / 1.055, 2.4);
			else G = G / 12.92;
			if (B > 0.04045) B = pow((B + 0.055) / 1.055, 2.4);
			else B = B / 12.92;

			double X = R * 0.4124 + G * 0.3576 + B * 0.1805;
			double Y = R * 0.2126 + G * 0.7152 + B * 0.0722;
			double Z = R * 0.0193 + G * 0.1192 + B * 0.9505;

			/*X = X / 95.047;
			Y = Y / 100.0;
			Z = Z / 108.883;

			double bounce = pow(6.0 / 29.0, 3);

			if (X > bounce)X = pow(X, 1.0/3.0);
			else X = 1.0 / 3.0*(29.0 / 6.0)*(29.0 / 6.0)*X + 4.0 / 29.0;
			if (Y > bounce)Y = pow(Y, 1.0 / 3.0);
			else Y = 1.0 / 3.0*(29.0 / 6.0)*(29.0 / 6.0)*Y + 4.0 / 29.0;
			if (Z > bounce)Z = pow(Z, 1.0 / 3.0);
			else Z = 1.0 / 3.0*(29.0 / 6.0)*(29.0 / 6.0)*Z + 4.0 / 29.0;
			*/
			const double epsilon = 0.008856;	//actual CIE standard
			const double kappa = 903.3;		//actual CIE standard

			const double Xr = 0.950456;	//reference white
			const double Yr = 1.0;		//reference white
			const double Zr = 1.088754;	//reference white
			X = X / Xr;
			Y = Y / Yr;
			Z = Z / Zr;

			if (X > epsilon)	X = pow(X, 1.0 / 3.0);
			else				X = (kappa*X + 16.0) / 116.0;
			if (Y > epsilon)	Y = pow(Y, 1.0 / 3.0);
			else				Y = (kappa*Y + 16.0) / 116.0;
			if (Z > epsilon)	Z = pow(Z, 1.0 / 3.0);
			else				Z = (kappa*Z + 16.0) / 116.0;


			lab_image[i][j] = new double[3];
			lab_image[i][j][0] = 116 * Y - 16;
			lab_image[i][j][1] = 500 * (X - Y);
			lab_image[i][j][2] = 200 * (Y - Z);
			//std::cout << rgb_image.ptr(0] << " " << rgb_image.ptr(1] << " " << rgb_image.ptr(2] << " " << std::endl;
		}
	}
	return lab_image;
}
//label is to log the pixel's label, 
//lenght is to tell how many pixel there is, 
//numk is to tell how many superpixels there are, 
//compactness is to tell whether the cluster is compactness, 
//least is to tell the mini size of the cluster, 
//the combine is to tell the threshold of the simiarity between pixel, 
//range is to tell the farthest pixel of the cluster can touch.
void IMAGE::SAM(std::vector<int>&label,int numk, double compactness, int least, double combine, double scope) {
	std::priority_queue< Cluster_pixel, std::vector< Cluster_pixel>, ::compare> q;
	std::vector< Cluster_pixel> clustercenter;
	std::vector<int> number;
	std::vector<int> seq;
	const int num = rgb_image.rows*rgb_image.cols;
	const int width = rgb_image.cols;
	const int height = rgb_image.rows;
	const double S=find_seed(numk, clustercenter);

	label.resize(num);
	for (int i = 0; i < num; ++i) {
		label[i] = -1;
	}

	number.resize(clustercenter.size());
	int  Cluster_pixelcount = 0;
	for (int i = 0; i < clustercenter.size(); ++i) {
		 Cluster_pixel t = clustercenter[i];
		clustercenter[i].p[0] = clustercenter[i].p[0] / combine;
		clustercenter[i].p[1] = clustercenter[i].p[1] / combine;
		clustercenter[i].p[2] = clustercenter[i].p[2] / combine;
		clustercenter[i].x = clustercenter[i].x / scope;
		clustercenter[i].y = clustercenter[i].y / scope;
		q.push(t);
		number[i] = 0;
	}
	//std::cout << clusteryenter.size() << std::endl;

	while (!q.empty()) {
		 Cluster_pixel temp = q.top();
		q.pop();
		int k = temp.label;
		int ind = index(temp.x, temp.y);
		//clusteryenter[k].upgrade(temp);
		if (label[ind] < 0) {
			label[ind] = k;
			seq.push_back(ind);
			++ Cluster_pixelcount;

			clustercenter[k].p[0] += temp.p[0];
			clustercenter[k].p[1] += temp.p[1];
			clustercenter[k].p[2] += temp.p[2];
			clustercenter[k].x += temp.x;
			clustercenter[k].y += temp.y;

			++number[k];

			for (int i = 0; i < CONNECTIVITY; ++i) {
				int x,y;
				x = temp.x + direction[i][0];
				y = temp.y + direction[i][1];
				if (!(x < 0 || x >= rgb_image.rows || y < 0 || y >= rgb_image.cols)) {
					int p = index(x, y);
					if (label[p] < 0) {
						 Cluster_pixel t(lab_image[x][y],x,y);

						/*// this method will result in more clusters
						double dp[3];
						double dx, dy;

						dp[0] = clustercenter[k].p[0] - t.p[0]*number[k];
						dp[1] = clustercenter[k].p[1] - t.p[1]*number[k];
						dp[2] = clustercenter[k].p[2] - t.p[2]*number[k];

						dx= clustercenter[k].x - t.x*number[k];
						dy= clustercenter[k].y - t.y*number[k];

						double dis = (dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2])
							+ (dx*dx + dy * dy)*(compactness*compactness) / (S);
						dis /= number[k];*/

						//this method will result in less clusters than the previous one
						double dp[3];
						double dx, dy;

						dp[0] = clustercenter[k].p[0] / number[k] - t.p[0];
						dp[1] = clustercenter[k].p[1] / number[k] - t.p[1];
						dp[2] = clustercenter[k].p[2] / number[k] - t.p[2];

						dx = clustercenter[k].x / number[k] - t.x;
						dy = clustercenter[k].y / number[k] - t.y;

						double dis = (dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2])
							+ (dx*dx + dy * dy)*(compactness*compactness) / (S);

						t.dis = dis;
						t.label = k;
						q.push(t);
					}
				}
			}
		}
	}

	for (int p = 0; p < num; ++p) {
			if (label[p] < 0) {
				if (p > 0 && label[p - 1] >= 0)label[p] = label[p - 1];
				else if (p < num - 1 && label[p + 1] >= 0)label[p] = label[p + 1];
			}
			else if (number[label[p]] <= least) {
				--number[label[p]];
				if (p > 0 && label[p - 1] != label[p]) {
					label[p] = label[p - 1];
					++number[label[p - 1]];
				}
				else if (p < num - 1 && label[p + 1] != label[p]) {
					label[p] = label[p + 1];
					++number[label[p + 1]];
				}
			}
	}
	add_bounce(label, 0xFF);
}
double IMAGE::find_seed(int& numk,std::vector< Cluster_pixel> &q) {
	const int sz = rgb_image.rows*rgb_image.cols;
	int gridstep = sqrt(double(sz) / double(numk)) + 0.5;
	int halfstep = gridstep / 2;
	double h = rgb_image.rows; double w = rgb_image.cols;

	int xsteps = int(w / gridstep);
	int ysteps = int(h / gridstep);
	int err1 = abs(xsteps*ysteps - numk);
	int err2 = abs(int(w / (gridstep - 1))*int(w / (gridstep - 1)) - numk);
	if (err2 < err1)
	{
		gridstep -= 1.0;
		xsteps = w / (gridstep);
		ysteps = h / (gridstep);
	}
	double S = gridstep * gridstep;
	int k = 0;
	for (int i = halfstep; i < h; i += gridstep) {
		for (int j = halfstep; j < w; j += gridstep) {
			 Cluster_pixel temp(lab_image[i][j],i,j);
			temp.dis = 0;
			temp.label = k++;
			q.push_back(temp);
		}
	}
	numk = q.size();
	return S;
	//_S = (double)_width * _height / numk;
}

void IMAGE::add_bounce(const std::vector<int>&label,uchar color,bool show) {
	const int width = rgb_image.cols;
	const int height = rgb_image.rows;
	const int num = width * height;
	cv::Mat image;
	rgb_image.copyTo(image);
	for (int i = 0; i < num; ++i) {
		int r = i / width;
		int c = i % width;
		for (int j = 0; j < 8; ++j) {
			int nr = r + direction[j][0];
			int nc = c + direction[j][1];
			if (nr >= 0 && nr < height&&nc >= 0 && nc < width) {
				if (label[i] != label[nr*width + nc])
				{
					uchar *p = image.ptr(r);
					p[c * 3] = color;//(uchar)(label[i]*2)&0x000000FF;
					p[c * 3 + 1] = color;//(uchar)((label[i]*5)&0x0000FF00)>>8;
					p[c * 3 + 2] = color;//(uchar)((label[i]*7)&0x00FF0000)>>16;
				}
			}
		}
	}
	if (show) {
		cv::namedWindow("SAM");
		cv::imshow("SAM", image);
		cv::waitKey();
		cv::destroyWindow("SAM");
	}
}


void IMAGE::MSAM() {
	std::thread LABEL(&IMAGE::get_label,this),REGION(&IMAGE::get_region,this);
	LABEL.join();
	REGION.join();
}

void IMAGE::get_region() {

	const int width = rgb_image.cols;
	const int height = rgb_image.rows;
	const int num = width * height;
	const int OBJECT_LABEL = -1;
	const int BACKGROUND_LABEL = -2;
	std::vector<int>label;


	SAM(label, 200, 1, 100, 5, 2);

	while (!done) {
		std::this_thread::yield();
	}
	done = false;
	std::unique_lock<std::mutex> lck(mtx);

	std::map<int,Region> region;
	
	int object_bounce = Object.size();
	int background_bounce = Background.size() + object_bounce;

	//for (int i = 0; i < Object.size(); ++i) {

	//	for (int j = 0; j < Object[i].size(); ++j) {
	//		int ind = index(Object[i][j].x, Object[i][j].y);
	//		if (check(Object[i][j].x, Object[i][j].y))
	//			label[ind] = -i-1;
	//		else int k = 0;
	//	}
	//	if(Object[i].size()>0)
	//		region.push_back(Region(Object[i]));
	//}

	//for (int i = 0; i < Background.size(); ++i) {	
	//	for (int j = 0; j < Background[i].size(); ++j) {
	//		int ind = index(Background[i][j].x, Background[i][j].y);
	//		if(check(Background[i][j].x,Background[i][j].y))
	//			label[ind] = -(i+object_bounce+1);
	//		else int k;
	//	}
	//	if (Background[i].size() > 0)
	//		region.push_back(Region(Background[i]));
	//}

	region[OBJECT_LABEL]=set_label_from_mark(Object, label, -1);
	region[BACKGROUND_LABEL]=set_label_from_mark(Background, label, -2);	

	add_bounce(label,0xFF,true);

	get_region_from_label(region, label);
	set_region_neighbour(region, label);
	return;
}

void IMAGE::get_label() {
	
	std::unique_lock<std::mutex> lck(mtx);
	cv::Mat image = rgb_image.clone();

	cv::namedWindow(name, CV_WINDOW_AUTOSIZE);
	cv::imshow(name, image);
	cv::setMouseCallback(name,on_mouse,(void*)&image);
	int c;
	do {
		c = cvWaitKey(0);
		switch ((char)c) {
		case 'g':
			flag = 0;
			break;
		case 'b':
			flag = 1;
			break;
		case 'e':
			break;
		case 's':
			cv::imwrite("G:/Matlab project/superpixel/bee_label.jpg", image);
			break;
		}
	} while (c != 115 && c != 101);
	done = true;
}
IMAGE::~IMAGE() {
	for (int i = 0;i< rgb_image.rows; ++i) {
		for (int j = 0;j< rgb_image.cols; ++j) {
			delete lab_image[i][j];
		}
		delete lab_image[i];
	}
	delete lab_image;
}

void IMAGE::get_region_from_label(std::map<int,Region>& region,const std::vector<int>& label) {
	const int width = rgb_image.cols;
	const int height = rgb_image.rows;
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			int ind = index(i, j);
			if (label[ind] >= 0) {
				region[label[ind]].add(Pixel(i, j));
			}
		}
	}
}

void IMAGE::set_region_neighbour(std::map<int, Region>&region,const std::vector<int>&label) {
	const int width = rgb_image.cols;
	const int height = rgb_image.rows;
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			for (int k = 0; k < CONNECTIVITY; ++k) {
				int x = i + direction[k][0];
				int y = j + direction[k][1];
				int ind = index(i, j);
				int nind = index(x, y);
				if (check(x, y) && label[ind] != label[nind]) {
					region[label[ind]].add(label[nind], &region[label[nind]]);
					region[label[nind]].add(label[ind], &region[label[ind]]);
				}
			}
		}
	}
}

void IMAGE::bfs(int x,int y,int source,int sign,std::vector<int>&label,Region& r) {
	std::queue<Pixel> q;
	q.push(Pixel(x, y));
	while (!q.empty()) {
		Pixel t = q.front();
		q.pop();
		for (int i = 0; i < CONNECTIVITY; ++i) {
			int tx = t.x + direction[i][0];
			int ty = t.y + direction[i][1];
			int ind = index(tx, ty);
			if (check(tx, ty) && label[ind] == source) {
				label[ind] = sign;
				r.add(Pixel(tx, ty));

				rgb_image.ptr(tx, ty)[0] = 0;
				rgb_image.ptr(tx, ty)[1] = 0;
				rgb_image.ptr(tx, ty)[2] = 0;//test code

				q.push(Pixel(tx, ty));
			}
		}
	}
}

Region IMAGE::set_label_from_mark(const std::vector<std::vector<Pixel>>& mark, std::vector<int>&label,int sign) {
	Region region;
	for (int i = 0; i < mark.size(); ++i) {
		for (int j = 0; j < mark[i].size(); ++j) {
			int ind = index(mark[i][j].x, mark[i][j].y);
			if (check(mark[i][j].x, mark[i][j].y) && label[ind] != sign)
				bfs(get_x(ind), get_y(ind), label[ind], sign, label, region);
		}
	}
	return region;
}

const int IMAGE::CONNECTIVITY=4;


Region::Region(const std::vector<Pixel>&source):region(source) {
}
Region::Region() {}
Region::Region(const Pixel&p) {
	region.push_back(p);
}
bool Region::add(const Pixel&p) {
	region.push_back(p);
	return true;
}
bool Region::add(int ind, const Region*r) {
	neigh[ind] = r;
	return true;
}
void Region::operator+=(const Region&r) {
	region.insert(region.end(), r.region.begin(), r.region.end());
	for (std::map<int, const Region*>::const_iterator it = r.neigh.begin(); it != r.neigh.end(); ++it) {
		neigh[it->first] = it->second;
	}
}

void on_mouse(int event, int x, int y, int flags, void* param)
{
	static bool s_bMouseLButtonDown = false;
	static cv::Point s_cvPrePoint = cv::Point(0, 0);

	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
		temp.clear();
		s_bMouseLButtonDown = true;
		s_cvPrePoint = cv::Point(x, y);
		break;

	case  CV_EVENT_LBUTTONUP:
		s_bMouseLButtonDown = false;
		if (0==flag ) {
			Object.push_back(temp);
		}
		else if(1==flag)Background.push_back(temp);
		break;

	case CV_EVENT_MOUSEMOVE:
		if (s_bMouseLButtonDown)
		{
			float dx, dy, ty, k;
			dx = y - s_cvPrePoint.y; dy = x-s_cvPrePoint.x;
			k = dy / dx;
			int x0, x1;
			if (y < s_cvPrePoint.y) {
				x0 = y;
				x1 = s_cvPrePoint.y;
				ty = x;
			}
			else {
				x1 = y;
				x0 = s_cvPrePoint.y;
				ty = s_cvPrePoint.x;
			}
			for (int i = x0; i < x1; i++) {
				temp.push_back(Pixel(i, (int)ty));
				ty += k;
			}

			cv::Point cvCurrPoint = cvPoint(x, y);
			
			if(0==flag)
				cv::line(*((cv::Mat*)param), s_cvPrePoint, cvCurrPoint, CV_RGB(0, 255,0), 3);
			else if (1 == flag) 
				cv::line(*((cv::Mat*)param), s_cvPrePoint, cvCurrPoint, CV_RGB(0, 0, 255), 3);
			s_cvPrePoint = cvCurrPoint;
			cv::imshow(name, *((cv::Mat*)param));
		}
		break;
	}
}