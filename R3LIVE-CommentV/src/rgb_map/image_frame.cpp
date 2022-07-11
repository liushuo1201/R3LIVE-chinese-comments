/* 
This code is the implementation of our paper "R3LIVE: A Robust, Real-time, RGB-colored, 
LiDAR-Inertial-Visual tightly-coupled state Estimation and mapping package".

Author: Jiarong Lin   < ziv.lin.ljr@gmail.com >

If you use any code of this repo in your academic research, please cite at least
one of our papers:
[1] Lin, Jiarong, and Fu Zhang. "R3LIVE: A Robust, Real-time, RGB-colored, 
    LiDAR-Inertial-Visual tightly-coupled state Estimation and mapping package." 
[2] Xu, Wei, et al. "Fast-lio2: Fast direct lidar-inertial odometry."
[3] Lin, Jiarong, et al. "R2LIVE: A Robust, Real-time, LiDAR-Inertial-Visual
     tightly-coupled state Estimator and mapping." 
[4] Xu, Wei, and Fu Zhang. "Fast-lio: A fast, robust lidar-inertial odometry 
    package by tightly-coupled iterated kalman filter."
[5] Cai, Yixi, Wei Xu, and Fu Zhang. "ikd-Tree: An Incremental KD Tree for 
    Robotic Applications."
[6] Lin, Jiarong, and Fu Zhang. "Loam-livox: A fast, robust, high-precision 
    LiDAR odometry and mapping package for LiDARs of small FoV."

For commercial use, please contact me < ziv.lin.ljr@gmail.com > and
Dr. Fu Zhang < fuzhang@hku.hk >.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from this
    software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.
*/
#include "image_frame.hpp"

Image_frame::Image_frame()
{
    m_gama_para( 0 ) = 1.0;
    m_gama_para( 1 ) = 0.0;
    m_pose_w2c_q.setIdentity();
    m_pose_w2c_t.setZero();
};

Image_frame::~Image_frame()
{
    release_image();
}

void Image_frame::release_image()
{
    m_raw_img.release();
    m_img.release();
    m_img_gray.release();
}

// 也是给图像设置pose，只是这里是c2w
void Image_frame::refresh_pose_for_projection()
{
    m_pose_c2w_q = m_pose_w2c_q.inverse();
    m_pose_c2w_t = -(m_pose_w2c_q.inverse() * m_pose_w2c_t);
    m_if_have_set_pose = 1; 
}

// 给Image_frame设置pose.w2c
void Image_frame::set_pose(const eigen_q &pose_w2c_q, const vec_3 &pose_w2c_t)
{
    m_pose_w2c_q = pose_w2c_q;
    m_pose_w2c_t = pose_w2c_t;
    refresh_pose_for_projection();
}

// 设置图像的idx 
int Image_frame::set_frame_idx(int frame_idx)
{
    m_frame_idx = frame_idx;
    return m_frame_idx;
}

// 设置内参
void Image_frame::set_intrinsic(Eigen::Matrix3d &camera_K)
{
    m_cam_K = camera_K;
    m_if_have_set_intrinsic = 1;
    fx = camera_K(0, 0);
    fy = camera_K(1, 1);
    cx = camera_K(0, 2);
    cy = camera_K(1, 2);
    m_gama_para(0) = 1.0;
    m_gama_para(1) = 0.0;
}

Image_frame::Image_frame(Eigen::Matrix3d &camera_K)
{
    m_pose_w2c_q.setIdentity();
    m_pose_w2c_t.setZero();
    set_intrinsic(camera_K);
};

// RGB转灰度图
void Image_frame::init_cubic_interpolation()
{
    m_pose_w2c_R = m_pose_w2c_q.toRotationMatrix();
    m_img_rows = m_img.rows;
    m_img_cols = m_img.cols;
#if (CV_MAJOR_VERSION >= 4)
    cv::cvtColor(m_img, m_img_gray, cv::COLOR_RGB2GRAY);
#else
    cv::cvtColor(m_img, m_img_gray, CV_RGB2GRAY);
#endif
}

// inverse_pose
void Image_frame::inverse_pose()
{
    m_pose_w2c_t = -(m_pose_w2c_q.inverse() * m_pose_w2c_t);
    m_pose_w2c_q = m_pose_w2c_q.inverse();
    m_pose_w2c_R = m_pose_w2c_q.toRotationMatrix();

    m_pose_c2w_q = m_pose_w2c_q.inverse();
    m_pose_c2w_t = -(m_pose_w2c_q.inverse() * m_pose_w2c_t);
}

/**
 * @brief 世界坐标系下的点投影到图像像素坐标uv
 * 
 * @param in_pt 世界坐标系下的3D点
 * @param cam_K 相机内参
 * @param u     投影到的像素坐标
 * @param v 
 * @param scale 图像缩放系数
 * @return true   图像坐标系下点深度满足要求
 * @return false  图像坐标系下点深度过近 < 0.001
 */
bool Image_frame::project_3d_to_2d(const pcl::PointXYZI & in_pt, Eigen::Matrix3d &cam_K, double &u, double &v, const double &scale)
{
    if (!m_if_have_set_pose)  // 当前图像没有pose，循环一直等待有pose为止
    {
        cout << ANSI_COLOR_RED_BOLD << "You have not set the camera pose yet!" << ANSI_COLOR_RESET << endl;
        // refresh_pose_for_projection();
        while (1)
        {};
    }
    if (m_if_have_set_intrinsic == 0) // 当前图像没有内参，循环一直等待有内参为止
    {
        cout << "You have not set the intrinsic yet!!!" << endl;
        while (1)
        {} ;
        return false;
    }

    vec_3 pt_w(in_pt.x, in_pt.y, in_pt.z), pt_cam;
    // pt_cam = (m_pose_w2c_q.inverse() * pt_w - m_pose_w2c_q.inverse()*m_pose_w2c_t);
    pt_cam = (m_pose_c2w_q * pt_w + m_pose_c2w_t); // 图像坐标系下的3d点
    if (pt_cam(2) < 0.001)   // 距离太近
    {
        return false;
    }
    u = (pt_cam(0) * fx / pt_cam(2) + cx) * scale;
    v = (pt_cam(1) * fy / pt_cam(2) + cy) * scale;
    return true;
}

/**
 * @brief     判断像素坐标是否在图像内，可以设置fov_mar来忽略一些在边沿的点，认为边沿点也不在图像内
 * 
 * @param u       像素坐标
 * @param v 
 * @param scale    缩放尺度
 * @param fov_mar  图像边沿的比例
 * @return true 
 * @return false 
 */
bool Image_frame::if_2d_points_available(const double &u, const double &v, const double &scale, double fov_mar)
{
    double used_fov_margin = m_fov_margin;
    if (fov_mar > 0.0)
    {
        used_fov_margin = fov_mar;
    }
    if ((u / scale >= (used_fov_margin * m_img_cols + 1)) && (std::ceil(u / scale) < ((1 - used_fov_margin) * m_img_cols)) &&
        (v / scale >= (used_fov_margin * m_img_rows + 1)) && (std::ceil(v / scale) < ((1 - used_fov_margin) * m_img_rows)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

/**
 * @brief 获取金子塔处的亚像素图像值
 * 
 * @tparam T 
 * @param mat              底层（原）图像
 * @param row              这里是（该层的）像素坐标！！！
 * @param col 
 * @param pyramid_layer    当前所处的金子塔层数
 * @return T               插值的亚像素位置的图像值
 */
template<typename T>
inline T getSubPixel(cv::Mat & mat, const double & row, const  double & col, double pyramid_layer = 0)
{
	int floor_row = floor(row);
	int floor_col = floor(col);
	double frac_row = row - floor_row; // 小数部分
	double frac_col = col - floor_col;
	int ceil_row = floor_row + 1;
	int ceil_col = floor_col + 1;
    // 计算在底层（原）图像的像素位置
    if (pyramid_layer != 0)  
    {
        int pos_bias = pow(2, pyramid_layer - 1);
        floor_row -= pos_bias;
        floor_col -= pos_bias;
        ceil_row += pos_bias;
        ceil_row += pos_bias;
    }
    // 双线性插值得到亚像素处的图像值
    return ((1.0 - frac_row) * (1.0 - frac_col) * (T)mat.ptr<T>(floor_row)[floor_col]) +
               (frac_row * (1.0 - frac_col) * (T)mat.ptr<T>(ceil_row)[floor_col]) +
               ((1.0 - frac_row) * frac_col * (T)mat.ptr<T>(floor_row)[ceil_col]) +
               (frac_row * frac_col * (T)mat.ptr<T>(ceil_row)[ceil_col]);
}

/**
 * @brief 
 * 
 * @param u      金字塔某一层的像素坐标
 * @param v 
 * @param layer  金字塔所处的层
 * @param rgb_dx 该像素左右两边的图像差
 * @param rgb_dy 该像素上下两边的图像差
 * @return vec_3 插值的亚像素位置的图像值
 */
vec_3 Image_frame::get_rgb(double &u, double v, int layer, vec_3 *rgb_dx , vec_3 *rgb_dy )
{
    cv::Vec3b rgb = getSubPixel< cv::Vec3b >( m_img, v, u, layer );
    if ( rgb_dx != nullptr )
    {
        cv::Vec3b rgb_left = getSubPixel< cv::Vec3b >( m_img, v, u - 1, layer );
        cv::Vec3b rgb_right = getSubPixel< cv::Vec3b >( m_img, v, u + 1, layer );
        cv::Vec3b cv_rgb_dx = rgb_right - rgb_left;
        *rgb_dx = vec_3( cv_rgb_dx( 0 ), cv_rgb_dx( 1 ), cv_rgb_dx( 2 ) );
    }
    if ( rgb_dy != nullptr )
    {
        cv::Vec3b rgb_down = getSubPixel< cv::Vec3b >( m_img, v - 1, u, layer );
        cv::Vec3b rgb_up = getSubPixel< cv::Vec3b >( m_img, v + 1, u, layer );
        cv::Vec3b cv_rgb_dy = rgb_up - rgb_down;
        *rgb_dy = vec_3( cv_rgb_dy( 0 ), cv_rgb_dy( 1 ), cv_rgb_dy( 2 ) );
    }
    return vec_3( rgb( 0 ), rgb( 1 ), rgb( 2 ) );
}

// 获得像素位置处的灰度值
double Image_frame::get_grey_color( double &u, double &v, int layer )
{
    double val = 0;

    if ( layer == 0 )
    {
        double gray_val = getSubPixel< uchar >( m_img, v, u );
        return gray_val;
    }
    else
    {
        // TODO
        while ( 1 )
        {
            cout << "To be process here" << __LINE__ << endl;
            std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
        };
    }

    // 到不了这里，所以没用
    return m_gama_para( 0 ) * val + m_gama_para( 1 );
}

// 获取位置处的RGB值
bool Image_frame::get_rgb(const double &u, const double &v, int &r, int &g, int &b)
{
    r = m_img.at<cv::Vec3b>(v, u)[2];
    g = m_img.at<cv::Vec3b>(v, u)[1];
    b = m_img.at<cv::Vec3b>(v, u)[0];
    return true;
}

// 打印一些信息。没什么用
void Image_frame::display_pose()
{
    cout << "Frm [" << m_frame_idx << "], pose: " << m_pose_w2c_q.coeffs().transpose() << " | " << m_pose_w2c_t.transpose() << " | ";
    cout << fx << ", " << cx << ", " << fy << ", " << cy << ", ";
    cout << endl;
}

// 图像直方图均衡（子函数）
void Image_frame::image_equalize(cv::Mat &img, int amp)
{
    cv::Mat img_temp;
    cv::Size eqa_img_size = cv::Size(std::max(img.cols * 32.0 / 640, 4.0), std::max(img.cols * 32.0 / 640, 4.0));
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(amp, eqa_img_size);
    // Equalize gray image.
    clahe->apply(img, img_temp);
    img = img_temp;
}

// 图像直方图均衡（子函数）
inline void image_equalize(cv::Mat &img, int amp)
{
    cv::Mat img_temp;
    cv::Size eqa_img_size = cv::Size(std::max(img.cols * 32.0 / 640, 4.0), std::max(img.cols * 32.0 / 640, 4.0));
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(amp, eqa_img_size);
    // Equalize gray image.
    clahe->apply(img, img_temp);
    img = img_temp;
}

// 图像直方图均衡（子函数）
inline cv::Mat equalize_color_image_Ycrcb(cv::Mat &image)
{
    cv::Mat hist_equalized_image;
    cv::cvtColor(image, hist_equalized_image, cv::COLOR_BGR2YCrCb);

    //Split the image into 3 channels; Y, Cr and Cb channels respectively and store it in a std::vector
    std::vector<cv::Mat> vec_channels;
    cv::split(hist_equalized_image, vec_channels);

    //Equalize the histogram of only the Y channel
    // cv::equalizeHist(vec_channels[0], vec_channels[0]);
    image_equalize( vec_channels[0], 1 );
    cv::merge(vec_channels, hist_equalized_image);
    cv::cvtColor(hist_equalized_image, hist_equalized_image, cv::COLOR_YCrCb2BGR);
    return hist_equalized_image;
}

// 图像直方图均衡
void Image_frame::image_equalize()
{
    image_equalize(m_img_gray, 3.0);
    // cv::imshow("before", m_img.clone());
    m_img = equalize_color_image_Ycrcb(m_img);
    // cv::imshow("After", m_img.clone());
}

/**
 * @brief 将3D点投影到图像内染色
 * 
 * @param in_pt               待投影的点
 * @param u                   投影到的像素坐标
 * @param v 
 * @param rgb_pt              投影染色后的点
 * @param intrinsic_scale     缩放尺度
 * @return true 
 * @return false 
 */
bool Image_frame::project_3d_point_in_this_img(const pcl::PointXYZI & in_pt, double &u, double &v, pcl::PointXYZRGB *rgb_pt, double intrinsic_scale)
{
    // 世界坐标系下的点投影到图像像素坐标uv
    if (project_3d_to_2d(in_pt, m_cam_K, u, v, intrinsic_scale) == false)
    {
        return false;
    }
    // 判断像素坐标是否在图像内
    if (if_2d_points_available(u, v, intrinsic_scale) == false)
    {
        // printf_line;
        return false;
    }
    if (rgb_pt != nullptr)
    {
        int r = 0;
        int g = 0;
        int b = 0;

        // 获取位置处的RGB值
        get_rgb(u, v, r, g, b); 

        // 为这个点染色
        rgb_pt->x = in_pt.x;
        rgb_pt->y = in_pt.y;
        rgb_pt->z = in_pt.z;
        rgb_pt->r = r;
        rgb_pt->g = g;
        rgb_pt->b = b;
        rgb_pt->a = 255;
    }
    return true;
}

// 将3D点投影到图像内染色
bool Image_frame::project_3d_point_in_this_img(const vec_3 & in_pt, double &u, double &v, pcl::PointXYZRGB *rgb_pt, double intrinsic_scale)
{
    pcl::PointXYZI temp_pt;
    temp_pt.x = in_pt(0);
    temp_pt.y = in_pt(1);
    temp_pt.z = in_pt(2);
    return project_3d_point_in_this_img(temp_pt, u, v, rgb_pt, intrinsic_scale);
}

// 把图像信息写入文件
void Image_frame::dump_pose_and_image(const std::string name_prefix)
{
    std::string txt_file_name = std::string(name_prefix).append(".txt");
    std::string image_file_name = std::string(name_prefix).append(".png");
    FILE *fp = fopen(txt_file_name.c_str(), "w+");
    if (fp)
    {
        fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf\r\n", m_pose_w2c_q.w(), m_pose_w2c_q.x(), m_pose_w2c_q.y(), m_pose_w2c_q.z(),
                m_pose_w2c_t(0), m_pose_w2c_t(1), m_pose_w2c_t(2));
        fclose(fp);
    }
    cv::imwrite(image_file_name, m_img);
}

// 从文件加载图像
int Image_frame::load_pose_and_image(const std::string name_prefix, const double image_scale, int if_load_image)
{
    // cout << "Load data from " << name_prefix << ".X" << endl;
    std::string txt_file_name = std::string(name_prefix).append(".txt");
    std::string image_file_name = std::string(name_prefix).append(".png");
    Eigen::MatrixXd pose_data = Common_tools::load_mat_from_txt<double>(txt_file_name);
    if (pose_data.size() == 0)
    {
        // cout << "Load offline data return fail." << endl;
        return 0;
    }
    // cout << "Pose data = " << pose_data << endl;
    m_pose_w2c_q = Eigen::Quaterniond(pose_data(0), pose_data(1), pose_data(2), pose_data(3));
    if (if_load_image)
    {
        m_img = cv::imread(image_file_name.c_str());
        if (image_scale != 1.0)
        {
            cv::resize(m_img, m_img, cv::Size(0, 0), image_scale, image_scale);
        }
        m_img_rows = m_img.rows;
        m_img_cols = m_img.cols;
    }
    // m_pose_w2c_q = Eigen::Map<Eigen::Quaterniond>(&pose_data.data()[0]);
    m_pose_w2c_R = m_pose_w2c_q.toRotationMatrix();
    m_pose_w2c_t = Eigen::Map<Eigen::Vector3d>(&pose_data.data()[4]);
    return 1;
}
