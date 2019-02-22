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
#include <condition_variable>
#include <atomic>
#include <future>
#define RGB_PIXEL false
#define LAB_PIXEL true

const int direction[8][2] = { {1,0}, {0,1},{-1,0},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1} };

class Pixel;
class Cluster_Pixel;
class compare;
class Region;
class IMAGE;

class Pixel {
public:
	friend class IMAGE;
	friend class compare;
	friend class Region;
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
	Region(IMAGE *,const std::vector< Pixel>&);
	Region(IMAGE *);
	Region(IMAGE *,const Pixel&);
	Region();
	Region(const Region&);
	bool add(const Pixel&);
	bool add(int,const Region*,double=1);
	void operator+=(const Region&);
	Region operator=(const Region&);
	std::vector<int> combine(const Region&);
	std::vector<int> combine(int);
	void Init_color_distance();
	size_t size() { return region.size(); }
	void set_image(IMAGE *const image) { this->image = image; }
	double get_real_distance(const int);
	const std::vector<Pixel>& get_region_pixel() const{ return region; }
private:
	const static int dimension;
	int find_color_id(const uchar *,const int,const int*);

	double(*kernel)(double);
	double color_dis[4096];
	IMAGE *image;
	std::vector<Pixel> region;
	std::map<int,const Region*> neigh;
	std::map<int, int> distance;
};

double kernel(double);


class IMAGE {
public:
	const static int CONNECTIVITY;
	IMAGE(const cv::Mat&source);
	//IMAGE(const int const*, const int const*, const int const*, int, int);
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
	uchar* get_rgb_pixel_color(int x, int y) {
		return rgb_image.ptr(x, y);
	}
	~IMAGE();
private:
	double find_seed(int&,std::vector< Cluster_pixel>&);
	int index(int x, int y) {
		return x * rgb_image.cols + y;
	}
	int get_x(int ind) {
		return ind / rgb_image.cols;
	}
	int get_y(int ind) {
		return ind % rgb_image.cols;
	}
	void add_bounce(const std::vector<int>&,uchar,bool = false);

	void get_label();
	void get_region(std::map<int,Region>&, std::vector<int>&);
	void combine(const std::map<int,Region>&,std::vector<int>&);

	bool check(int x, int y) {
		if (x >= 0 && x < rgb_image.rows&&y >= 0 && y < rgb_image.cols)
			return true;
		return false;
	}

	Region set_label_from_mark(const std::vector<std::vector<Pixel>>& mark, std::vector<int>&label, int sign);
	void get_region_from_label(std::map<int,Region>&,const std::vector<int>&);
	void set_region_neighbour(std::map<int,Region>&,const std::vector<int>&);
	void set_label_from_region(std::vector<int>&, const Region&,int);
	void bfs(int,int,int,int,std::vector<int>&,Region&);

	cv::Mat rgb_image;
	double*** lab_image;
	std::mutex mtx;
	std::atomic<bool> done;
};

void on_mouse(int, int, int, int, void*);
void show_histogram(const double*,const int);
#endif