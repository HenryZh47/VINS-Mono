#include "feature_manager.h"
#include <boost/math/distributions/normal.hpp>
#include <vector>

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {

        it.used_num = it.feature_per_frame.size();

        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}


bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;

    double depth_mean, depth_min;
    bool is_valid_prev_depth = getFrameDepthInfo(frame_count, depth_mean, depth_min);

    for (auto &id_pts : image)
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

        int feature_id = id_pts.first;
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });

        if (it == feature.end())
        {
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
        }

        // henryzh47: also initialize feature depth filter properties
        if (is_valid_prev_depth) {
            if (it !=feature.end() && !it->df_initialized && it->solve_flag == 1) {
                dfInit(it->estimated_depth, depth_min, *it);
            }
        }
    }

    if (frame_count < 2 || last_track_num < 20)
        return true;

    for (auto &it_per_id : feature)
    {
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

// TODO (henryzh47): test triangulate uncertainty
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        if (it_per_id.estimated_depth > 0)
            continue;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;

            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method;
        //it_per_id->estimated_depth = INIT_DEPTH;

        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }

    }
}

void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;

                // henryzh47: also shift depth filter points
                if (it->df_initialized) {
                    Eigen::Vector3d pts_i = uv_i * (1.0/it->mu);
                    Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                    Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                    double dep_j = pts_j(2);
                    if (dep_j > 0)
                        it->mu = 1.0/dep_j;
                    else
                        it->mu = 1.0/INIT_DEPTH;
                }
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}

// henryzh47: depth filter functions
bool FeatureManager::getFrameDepthInfo(int frame_count, double &depth_mean, double &depth_min) {
    if (frame_count < 2) {
        ROS_WARN("Not enough frames to get depth info for detph filter initialization");
        return false;
    }

    size_t frame_feature_count = 0;
    double depth_sum = 0.0;
    depth_min = std::numeric_limits<double>::max();
    for (auto &it_per_id : feature) {
        if (it_per_id.solve_flag == 1) {
            // valid depth
            depth_sum += it_per_id.estimated_depth;
            depth_min = fmin(it_per_id.estimated_depth, depth_min);
            frame_feature_count++;
        }
    }

    if (frame_feature_count == 0) {
        ROS_WARN("Cannot get scene depth from previous frame during depth filter initialization");
        return false;
    }
    depth_mean = depth_sum / frame_feature_count;
    return true;
}

void FeatureManager::setDepth(const VectorXd &x, Vector3d Ps[], Vector3d tic[], Matrix3d ric[]) {
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
            // TODO (henryzh47): here make feature uncertianty big
        }
        else
        {
            it_per_id.solve_flag = 1;

            // henryzh47: update feature depth filter
            if (it_per_id.df_initialized) {
                double z = it_per_id.estimated_depth;
                // double tau = dfComputeTau(it_per_id, Ps, tic, ric);
                double tau = 0.5;
                double tau_inv = 0.5 * (1.0/max(0.0000001, z-tau) - 1.0/(z+tau));
                dfUpdateSeed(1.0/z, tau_inv*tau_inv, it_per_id);
            }
        }
    }
}

double FeatureManager::dfComputeTau(FeaturePerId &f_per_id, Vector3d Ps[], Vector3d tic[], Matrix3d ric[]) {
    double fx = FOCAL_LENGTH;
    double px_error_angle = atan(PX_NOISE/(2.0*fx))*2.0;

    // get transform from first feature frame to latest feature frame
    int ref_i = f_per_id.start_frame;
    int cur_i = f_per_id.endFrame();
    Eigen::Vector3d ref_t = Ps[ref_i] + Rs[ref_i] * tic[0];
    Eigen::Matrix3d ref_R = Rs[ref_i] * ric[0];

    Eigen::Vector3d cur_t = Ps[cur_i] + Rs[cur_i] * tic[0];
    // Eigen::Matrix3d cur_R = Rs[cur_i] * ric[0];

    Eigen::Vector3d t = ref_R.transpose() * (cur_t - ref_t);
    // Eigen::Matrix3d R = ref_R.transpose() * cur_R;

    // use law of chord to compute 1 pixel error -> depth error
    Eigen::Vector3d f = f_per_id.feature_per_frame[0].point;
    double z = f_per_id.estimated_depth;
    Eigen::Vector3d a = f * z - t;
    double t_norm = t.norm();
    double a_norm = a.norm();

    double alpha = acos(f.dot(t)/t_norm); // dot product
    double beta = acos(a.dot(-t)/(t_norm*a_norm)); // dot product
    double beta_plus = beta + px_error_angle;
    double gamma_plus = PI-alpha-beta_plus; // triangle angles sum to PI
    double z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines

    ROS_DEBUG_STREAM("depth filter computed tau for feature: " << f_per_id.feature_id << ", tau: " << (z_plus - z));
    return (z_plus - z); // tau
}

void FeatureManager::dfInit(const double &depth_mean, const double &depth_min, FeaturePerId &f_per_id) {
    f_per_id.a = 10.0;
    f_per_id.b = 10.0;
    f_per_id.mu = 1.0 / depth_mean;
    // f_per_id.z_range = 1.0 / depth_min;
    f_per_id.z_range = 1.0 / 1.0;
    f_per_id.sigma2 = f_per_id.z_range * f_per_id.z_range / 36;
    f_per_id.df_initialized = true;
    ROS_DEBUG_STREAM("depth filter initialized feature: " << f_per_id.feature_id << ", depth filter to depth: "
                     << depth_mean << ", prev depth min: " << depth_min);
}

void FeatureManager::dfUpdateSeed(const double inv_depth, const double tau2, FeaturePerId &f_per_id) {
    double norm_scale = sqrt(f_per_id.sigma2 + tau2);
    if (std::isnan(norm_scale)) return;
    boost::math::normal_distribution<double> nd(f_per_id.mu, norm_scale);
    double s2 = 1./(1./f_per_id.sigma2 + 1./tau2);
    double m = s2 * (f_per_id.mu/f_per_id.sigma2 + inv_depth/tau2);

    double C1 = f_per_id.a / (f_per_id.a + f_per_id.b) * boost::math::pdf(nd, inv_depth);
    double C2 = f_per_id.b / (f_per_id.a + f_per_id.b) * 1./f_per_id.z_range;

    double normalized_constant = C1 + C2;
    C1 /= normalized_constant;
    C2 /= normalized_constant;

    double f = C1 * (f_per_id.a + 1.) / (f_per_id.a + f_per_id.b + 1.) + C2 * f_per_id.a / (f_per_id.a + f_per_id.b + 1.);
    double e = C1*(f_per_id.a+1.)*(f_per_id.a+2.)/((f_per_id.a+f_per_id.b+1.)*(f_per_id.a+f_per_id.b+2.))
               + C2*f_per_id.a*(f_per_id.a+1.0f)/((f_per_id.a+f_per_id.b+1.0f)*(f_per_id.a+f_per_id.b+2.0f));

    // update
    double mu_new = C1*m + C2*f_per_id.mu;
    f_per_id.sigma2 = C1*(s2+m*m) + C2*(f_per_id.sigma2 + f_per_id.mu*f_per_id.mu) - mu_new*mu_new;
    f_per_id.mu = mu_new;
    f_per_id.a = (e-f)/(f-e/f);
    f_per_id.b = f_per_id.a * (1.0-f)/f;

    ROS_DEBUG_STREAM("depth filter updated feature: " << f_per_id.feature_id << ", depth: " << 1.0 / f_per_id.mu
                     << ", sigma2: " << f_per_id.sigma2 << ", a: " << f_per_id.a << ", b: " << f_per_id.b);
}