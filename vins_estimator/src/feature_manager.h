#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"

class FeaturePerFrame
{
  public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5); 
        velocity.y() = _point(6); 
        cur_td = td;
    }
    double cur_td;
    Vector3d point;
    Vector2d uv;
    Vector2d velocity;
    double z;
    bool is_used;
    double parallax;
    MatrixXd A;
    VectorXd b;
    double dep_gradient;
};

class FeaturePerId
{
  public:
    const int feature_id;
    int start_frame;
    vector<FeaturePerFrame> feature_per_frame;

    int used_num;
    bool is_outlier;
    bool is_margin;
    double estimated_depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    // henryzh47: depth filter for depth uncertainty
    bool df_initialized;
    double mu;      // mean of normal distribution (inverse depth)
    double sigma2;  // variance of normal distribution
    double a;        // a of Beta distribution: When high, probability of inlier is large.
    double b;        // b of Beta distribution: When high, probability of outlier is large.
    double z_range;  // 1/zmin

    Vector3d gt_p;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0),
          df_initialized(false), mu(-1.0), sigma2(2.0), a(10), b(10), z_range(10)
    {
    }

    int endFrame();
};

class FeatureManager
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);

    void clearState();

    int getFeatureCount();

    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    void debugShow();
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    void removeFailures();
    void clearDepth(const VectorXd &x);
    VectorXd getDepthVector();
    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier();
    list<FeaturePerId> feature;
    int last_track_num;

    // henryzh47: depth filtering
    // get feature depth info from previous frame to initialize depth filter properties for current frame
    bool getFrameDepthInfo(int frame_count, double &depth_mean, double &depth_min);
    // overload method for depth filtering
    void setDepth(const VectorXd &x, Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    // compute depth measurement uncertainty with camera pose and px error angle
    double dfComputeTau(FeaturePerId &f_per_id, Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    // initialize depth filter properties
    void dfInit(const double &depth_mean, const double &depth_min, FeaturePerId &f_per_id);
    // update feature depth seed
    void dfUpdateSeed(const double inv_depth, const double tau2, FeaturePerId &f_per_id);

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    const Matrix3d *Rs;
    Matrix3d ric[NUM_OF_CAM];

    // henryzh47: noise for feature tracking measurement
    const double PX_NOISE = 1.0;
    const double PI = 3.14159265;
};

#endif