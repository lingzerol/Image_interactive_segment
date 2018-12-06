#pragma once
#ifndef IMAGE_DEBUG
#define IMAGE_DEBUG
#include <core.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <queue>
#include <map>
#include <mutex>
#define RGB_PIXEL false
#define LAB_PIXEL true

const int direction[8][2] = { {1,0}, {0,1},{-1,0},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1} };

class Pixel {
public:
	friend class IMAGE;
	friend class compare;
	Pixel() :x(-1), y(-1) {}
	Pixel(long long _x, long long _y) : x(_x), y(_y){}
private:
	long long x, y;
};

class Cluster_pixel {
public:
	friend class IMAGE;
	friend class compare;
	Cluster_pixel() :label(-1), x(-1), y(-1), dis(DBL_MAX) {}
	Cluster_pixel(double a, double b, double c, long long _x, long long _y) :p{a,b,c},label(-1), x(_x), y(_y), dis(DBL_MAX) {}
	Cluster_pixel(double *_p, long long _x, long long _y) :label(-1), x(_x), y(_y), dis(DBL_MAX) {
		p[0] = _p[0];
		p[1] = _p[1];	
		p[2] = _p[2];
	}
private:
	long long label;
	long long x, y;
	double dis;
	double p[3];
};

class compare {
public:
	bool operator ()(const  Cluster_pixel&a, const  Cluster_pixel&b) {
		return a.dis > b.dis;
	}
};

class Region {
public:
	Region(const std::vector< Pixel>&);
	Region();
	Region(const Pixel&);
	bool add(const Pixel&);
	bool add(int,const Region*);
	void operator+=(const Region&);
	size_t size() { return region.size(); }
private:
	std::vector<Pixel> region;
	std::map<int,const Region*> neigh;
};


class IMAGE {
public:
	const static int CONNECTIVITY;
	IMAGE(const cv::Mat&source);
	IMAGE(const int const*, const int const*, const int const*, int, int);
	int width() { return rgb_image.rows; }
	int height() { return rgb_image.rows; }
	double*** rgb_to_lab();

	void SAM(std::vector<int>&,int=100, double=10, int=100, double=2, double=1);
	void MSAM();

	uchar* get_rgb(const  Cluster_pixel& p) {
		return rgb_image.ptr(p.x, p.y);
	}
	double* get_lab(const  Cluster_pixel& p) {
		return lab_image[p.x][p.y];
	}
	const uchar* get_rgb(const  Cluster_pixel& p) const{
		return rgb_image.ptr(p.x, p.y);
	}
	const double* get_lab(const  Cluster_pixel& p) const{
		return lab_image[p.x][p.y];
	}
	~IMAGE();
private:
	double find_seed(int&,std::vector< Cluster_pixel>&);
	int index(int x, int y) {
		return x * rgb_image.cols + y;
	}
	void add_bounce(const std::vector<int>&,uchar,bool = false);

	void get_label();
	void get_region();

	bool check(int x, int y) {
		if (x >= 0 && x < rgb_image.rows&&y >= 0 && y < rgb_image.cols)
			return true;
		return false;
	}

	void get_region_from_label(std::vector<Region>&, std::vector<int>&);
	void bfs(int,int,int,int,std::vector<int>&,Region&);

	cv::Mat rgb_image;
	double*** lab_image;
	std::mutex mtx;
};
void on_mouse(int, int, int, int, void*);
#endif