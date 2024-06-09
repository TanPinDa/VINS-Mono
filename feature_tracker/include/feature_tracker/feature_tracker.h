#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "feature_tracker/parameters.h"
#include "feature_tracker/tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt, const int &image_width, const int &image_height, const int border_size);

void FilterPoints(vector<cv::Point2f> &v, vector<uchar> status);
void FilterFeatureIds(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img,double _cur_time,const bool &detect_new_feature_points);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    void undistortedPoints();

    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat  cur_img, forw_img;
    vector<cv::Point2f> current_points, forw_pts;
    vector<cv::Point2f> previous_undistorted_points, current_undistorted_points;
    vector<cv::Point2f> pts_velocity;
    vector<int> feature_ids;
    vector<int> track_cnt;
    map<int, cv::Point2f> current_undistorted_points_by_id;
    map<int, cv::Point2f> previous_undistorted_points_by_id;
    camodocal::CameraPtr m_camera;
    double cur_time;
    double prev_time;

    static int n_id;
};
