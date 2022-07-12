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
#include "pointcloud_rgbd.hpp"
#include "../optical_flow/lkpyramid.hpp"
extern Common_tools::Cost_time_logger g_cost_time_logger;
extern std::shared_ptr<Common_tools::ThreadPool> m_thread_pool_ptr;
cv::RNG g_rng = cv::RNG(0);
// std::atomic<long> g_pts_index(0);

// 设置RGB_Point的xyz坐标
void RGB_pts::set_pos(const vec_3 &pos)
{
    m_pos[0] = pos(0);
    m_pos[1] = pos(1);
    m_pos[2] = pos(2);
}

// 这个点在世界坐标系下的xyz
vec_3 RGB_pts::get_pos()
{
    return vec_3(m_pos[0], m_pos[1], m_pos[2]);
}

mat_3_3 RGB_pts::get_rgb_cov()
{
    mat_3_3 cov_mat = mat_3_3::Zero();
    for (int i = 0; i < 3; i++)
    {
        cov_mat(i, i) = m_cov_rgb[i];
    }
    return cov_mat;
}

vec_3 RGB_pts::get_rgb()
{
    return vec_3(m_rgb[0], m_rgb[1], m_rgb[2]);
}

pcl::PointXYZI RGB_pts::get_pt()
{
    pcl::PointXYZI pt;
    pt.x = m_pos[0];
    pt.y = m_pos[1];
    pt.z = m_pos[2];
    return pt;
}

void RGB_pts::update_gray(const double gray, const double obs_dis)
{
    if (m_obs_dis != 0 && (obs_dis > m_obs_dis * 1.2))
    {
        return;
    }
    m_gray = (m_gray * m_N_gray + gray) / (m_N_gray + 1);
    if (m_obs_dis == 0 || (obs_dis < m_obs_dis))
    {
        m_obs_dis = obs_dis;
        // m_gray = gray;
    }
    m_N_gray++;
    // TODO: cov update
};

const double image_obs_cov = 15;
const double process_noise_sigma = 0.1;

/**
 * @brief 
 * 
 * @param rgb     这个点的RGB颜色
 * @param obs_dis  观测到这个点的相机到这个点的距离
 * @param obs_sigma  观测方差
 * @param obs_time   观测到这个点的相机的时间
 * @return int  0 没有更新 1更新
 */
int RGB_pts::update_rgb(const vec_3 &rgb, const double obs_dis, const vec_3 obs_sigma, const double obs_time)
{
    // 距离过大，放弃更新
    if (m_obs_dis != 0 && (obs_dis > m_obs_dis * 1.2))
    {
        return 0;
    }

    // 第一次被观测
    if( m_N_rgb == 0)
    {
        // For first time of observation.
        m_last_obs_time = obs_time;  // 时间
        m_obs_dis = obs_dis; // 距离
        
        for (int i = 0; i < 3; i++)
        {
            m_rgb[i] = rgb[i]; // 颜色
            m_cov_rgb[i] = obs_sigma(i) ; // 方差
        }
        m_N_rgb = 1;
        return 0;
    }
    // State estimation for robotics, section 2.2.6, page 37-38
    // 不是第一次观测
    for(int i = 0 ; i < 3; i++)
    {
        // 更新方差
        m_cov_rgb[i] = (m_cov_rgb[i] + process_noise_sigma * (obs_time - m_last_obs_time)); // Add process noise        
        double old_sigma = m_cov_rgb[i];
        m_cov_rgb[i] = sqrt( 1.0 / (1.0 / m_cov_rgb[i] / m_cov_rgb[i] + 1.0 / obs_sigma(i) / obs_sigma(i)) );

        // 更新RGB，现在的RGB和原来的RGB融合
        m_rgb[i] = m_cov_rgb[i] * m_cov_rgb[i] * ( m_rgb[i] / old_sigma / old_sigma + rgb(i) / obs_sigma(i) / obs_sigma(i) );
    }

    // 更新观测到这个点的最远距离
    if (obs_dis < m_obs_dis)
    {
        m_obs_dis = obs_dis;
    }
    m_last_obs_time = obs_time; //更新上一次观测时间
    m_N_rgb++; // 更新观测到这个点的图像数
    return 1;
}

void Global_map::clear()
{
    m_rgb_pts_vec.clear();
}

void Global_map::set_minmum_dis(double minimum_dis)
{
    m_hashmap_3d_pts.clear();
    m_minimum_pts_size = minimum_dis;
}

Global_map::Global_map( int if_start_service )
{
    m_mutex_pts_vec = std::make_shared< std::mutex >();
    m_mutex_img_pose_for_projection = std::make_shared< std::mutex >();
    m_mutex_recent_added_list = std::make_shared< std::mutex >();
    m_mutex_rgb_pts_in_recent_hitted_boxes = std::make_shared< std::mutex >();
    m_mutex_m_box_recent_hitted = std::make_shared< std::mutex >();
    m_mutex_pts_last_visited = std::make_shared< std::mutex >();
    // Allocate memory for pointclouds
    if ( Common_tools::get_total_phy_RAM_size_in_GB() < 12 )
    {
        scope_color( ANSI_COLOR_RED_BOLD );
        std::this_thread::sleep_for( std::chrono::seconds( 1 ) );
        cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
        cout << "I have detected your physical memory smaller than 12GB (currently: " << Common_tools::get_total_phy_RAM_size_in_GB()
             << "GB). I recommend you to add more physical memory for improving the overall performance of R3LIVE." << endl;
        cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
        std::this_thread::sleep_for( std::chrono::seconds( 5 ) );
        m_rgb_pts_vec.reserve( 1e8 );
    }
    else
    {
        m_rgb_pts_vec.reserve( 1e9 );
    }
    // m_rgb_pts_in_recent_visited_voxels.reserve( 1e6 );
    if ( if_start_service )
    {
        m_thread_service = std::make_shared< std::thread >( &Global_map::service_refresh_pts_for_projection, this );
    }
}
Global_map::~Global_map(){};

//更新全局地图中用于投影的地图点和图像id，last_pose_q
void Global_map::service_refresh_pts_for_projection()
{
    eigen_q last_pose_q = eigen_q::Identity();
    Common_tools::Timer                timer;
    std::shared_ptr< Image_frame > img_for_projection = std::make_shared< Image_frame >();//指向当前用于渲染的图像
    while (1)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        m_mutex_img_pose_for_projection->lock();
         
        *img_for_projection = m_img_for_projection;//当前用于渲染的图像
        m_mutex_img_pose_for_projection->unlock();
        if (img_for_projection->m_img_cols == 0 || img_for_projection->m_img_rows == 0) //图像为空跳过
        {
            continue;
        }

        if (img_for_projection->m_frame_idx == m_updated_frame_index) //没有新的图像跳过（通过判断这个frame idx有没有更新）
        {
            continue;
        }
        timer.tic(" ");
        std::shared_ptr<std::vector<std::shared_ptr<RGB_pts>>> pts_rgb_vec_for_projection = std::make_shared<std::vector<std::shared_ptr<RGB_pts>>>();
        if (m_if_get_all_pts_in_boxes_using_mp)  // 初始化为1，在vio中设为0然后一直保持不变
        {
            std::vector<std::shared_ptr<RGB_pts>>  pts_in_recent_hitted_boxes;
            pts_in_recent_hitted_boxes.reserve(1e6);
            std::unordered_set< std::shared_ptr< RGB_Voxel> > boxes_recent_hitted;
            m_mutex_m_box_recent_hitted->lock();
            boxes_recent_hitted = m_voxels_recent_visited;
            m_mutex_m_box_recent_hitted->unlock();

            // get_all_pts_in_boxes(boxes_recent_hitted, pts_in_recent_hitted_boxes);
            m_mutex_rgb_pts_in_recent_hitted_boxes->lock();
            // m_rgb_pts_in_recent_visited_voxels = pts_in_recent_hitted_boxes;
            m_mutex_rgb_pts_in_recent_hitted_boxes->unlock();
        }
        //将上个scan的点投影到当前相机坐标系下，去掉投影过程中的重复点和遮挡点
        selection_points_for_projection(img_for_projection, pts_rgb_vec_for_projection.get(), nullptr, 10.0, 1);
        m_mutex_pts_vec->lock();
        m_pts_rgb_vec_for_projection = pts_rgb_vec_for_projection;//当前用于投影的点云
        m_updated_frame_index = img_for_projection->m_frame_idx;// 更新图像id
        // cout << ANSI_COLOR_MAGENTA_BOLD << "Refresh pts_for_projection size = " << m_pts_rgb_vec_for_projection->size()
        //      << " | " << m_rgb_pts_vec.size()
        //      << ", cost time = " << timer.toc() << ANSI_COLOR_RESET << endl;
        m_mutex_pts_vec->unlock();
        last_pose_q = img_for_projection->m_pose_w2c_q;
    }
}

// 没有用
void Global_map::render_points_for_projection(std::shared_ptr<Image_frame> &img_ptr)
{
    m_mutex_pts_vec->lock();
    if (m_pts_rgb_vec_for_projection != nullptr)
    {
        render_pts_in_voxels(img_ptr, *m_pts_rgb_vec_for_projection);
        // render_pts_in_voxels(img_ptr, m_rgb_pts_vec);
    }
    m_last_updated_frame_idx = img_ptr->m_frame_idx;
    m_mutex_pts_vec->unlock();
}

//
void Global_map::update_pose_for_projection(std::shared_ptr<Image_frame> &img, double fov_margin)
{
    m_mutex_img_pose_for_projection->lock();
    m_img_for_projection.set_intrinsic(img->m_cam_K);
    m_img_for_projection.m_img_cols = img->m_img_cols;
    m_img_for_projection.m_img_rows = img->m_img_rows;
    m_img_for_projection.m_fov_margin = fov_margin; // 边界
    m_img_for_projection.m_frame_idx = img->m_frame_idx;
    m_img_for_projection.m_pose_w2c_q = img->m_pose_w2c_q;  // 设置位姿
    m_img_for_projection.m_pose_w2c_t = img->m_pose_w2c_t;
    m_img_for_projection.m_img_gray = img->m_img_gray; // clone?
    m_img_for_projection.m_img = img->m_img;           // clone?
    m_img_for_projection.refresh_pose_for_projection();
    m_mutex_img_pose_for_projection->unlock();
}

// lio线程是否正在往全局地图添加点 1表示正在添加 0 表示结束
bool Global_map::is_busy()
{
    return m_in_appending_pts;
}

template int Global_map::append_points_to_global_map<pcl::PointXYZI>(pcl::PointCloud<pcl::PointXYZI> &pc_in, double  added_time, std::vector<std::shared_ptr<RGB_pts>> *pts_added_vec, int step);
template int Global_map::append_points_to_global_map<pcl::PointXYZRGB>(pcl::PointCloud<pcl::PointXYZRGB> &pc_in, double  added_time, std::vector<std::shared_ptr<RGB_pts>> *pts_added_vec, int step);

/**
 * @brief   把当前scan添加到全局地图    voxel = box 两者含义相同
 * 
 * @tparam T 
 * @param pc_in         全局坐标系下的当前scan的点
 * @param added_time    这个点被添加的时间。相对于第一个imu的时间：Measures.lidar_end_time - g_camera_lidar_queue.m_first_imu_time  
 * @param pts_added_vec pc_in落在的grid的集合（有重复）
 * @param step          降采样步长
 * @return int          返回这个scan落在了多少box
 */
template <typename T>
int Global_map::append_points_to_global_map(pcl::PointCloud<T> &pc_in, double  added_time,  std::vector<std::shared_ptr<RGB_pts>> *pts_added_vec, int step)
{
    m_in_appending_pts = 1; // lio线程是否正在往全局地图添加点 1表示正在添加 0 表示结束
    Common_tools::Timer tim;
    tim.tic();
    int acc = 0; // 接受的点数（不重复的点）
    int rej = 0; // 拒绝的点数（重复点）
    // 先清理，开始新一次处理
    if (pts_added_vec != nullptr)
    {
        pts_added_vec->clear();
    }
    std::unordered_set< std::shared_ptr< RGB_Voxel > > voxels_recent_visited; // 当前scan的点落在的box集合（会有重复的box）
    if (m_recent_visited_voxel_activated_time == 0) //  m_recent_visited_voxel_activated_time一直是0，没找见改变的地方（初始化是多少就一直是多少）
    {
        voxels_recent_visited.clear();
    }
    else // 没有使用
    {
        m_mutex_m_box_recent_hitted->lock();
        voxels_recent_visited = m_voxels_recent_visited;  // 用上一次的初始化这一次的
        m_mutex_m_box_recent_hitted->unlock();
        // 遍历
        for( Voxel_set_iterator it = voxels_recent_visited.begin(); it != voxels_recent_visited.end();  )
        {
            // 这个box距离上一次被添加已经过了很久
            if ( added_time - ( *it )->m_last_visited_time > m_recent_visited_voxel_activated_time )
            {
                it = voxels_recent_visited.erase( it ); // 删除这个box
                continue;
            }
            it++;
        }
        cout << "Restored voxel number = " << voxels_recent_visited.size() << endl;
    }


    int number_of_voxels_before_add = voxels_recent_visited.size();
    int pt_size = pc_in.points.size();
    // step = 4;
    // 遍历输入点云
    for (int pt_idx = 0; pt_idx < pt_size; pt_idx += step)
    {
        int add = 1; // 判断是否是重复点。当是重复点，add = 0

        // 计算grid索引
        // 小于m_minimum_pts_size的点被当作同一个点(05  9:13)
        int grid_x = std::round(pc_in.points[pt_idx].x / m_minimum_pts_size);
        int grid_y = std::round(pc_in.points[pt_idx].y / m_minimum_pts_size);
        int grid_z = std::round(pc_in.points[pt_idx].z / m_minimum_pts_size);

        // 计算box索引
        int box_x =  std::round(pc_in.points[pt_idx].x / m_voxel_resolution);
        int box_y =  std::round(pc_in.points[pt_idx].y / m_voxel_resolution);
        int box_z =  std::round(pc_in.points[pt_idx].z / m_voxel_resolution);

        
        if (m_hashmap_3d_pts.if_exist(grid_x, grid_y, grid_z))// 判断是否已经有这个grid了 
        {
            // 目前的条件相当于：取出有重复点的grid。
            add = 0;
            if (pts_added_vec != nullptr)
            {
                // 之前已经添加过的点的grid集合
                pts_added_vec->push_back(m_hashmap_3d_pts.m_map_3d_hash_map[grid_x][grid_y][grid_z]);
            }
        }

        RGB_voxel_ptr box_ptr;  // 该点所在的box
        if(!m_hashmap_voxels.if_exist(box_x, box_y, box_z)) // 判断是否已经有这个box了， 没有就加
        {
            std::shared_ptr<RGB_Voxel> box_rgb = std::make_shared<RGB_Voxel>();
            m_hashmap_voxels.insert( box_x, box_y, box_z, box_rgb );
            box_ptr = box_rgb;
        }
        else // 有就直接取
        {
            box_ptr = m_hashmap_voxels.m_map_3d_hash_map[box_x][box_y][box_z];
        }
        voxels_recent_visited.insert( box_ptr );  
        box_ptr->m_last_visited_time = added_time; // 这个box最近一次被添加的时间
        if (add == 0) // 如果是重复grid，跳过。避免重复
        {
            rej++;
            continue;
        }
        acc++;
        std::shared_ptr<RGB_pts> pt_rgb = std::make_shared<RGB_pts>(); // 构建一个RGB_pts格式的point
        pt_rgb->set_pos(vec_3(pc_in.points[pt_idx].x, pc_in.points[pt_idx].y, pc_in.points[pt_idx].z)); // 设置这个点的xyz位置

        pt_rgb->m_pt_index = m_rgb_pts_vec.size(); // 设置这个点的index。 因为size总比index多1，所以可以用这种方式给新点做索引
        m_rgb_pts_vec.push_back(pt_rgb);
        m_hashmap_3d_pts.insert(grid_x, grid_y, grid_z, pt_rgb); // 没有重复的
        box_ptr->add_pt(pt_rgb); // 往对应的box中插入点
        if (pts_added_vec != nullptr)  // TODO 又加一遍？那就相当于所有点都加了
        {
            pts_added_vec->push_back(pt_rgb);
        }
    }
    m_in_appending_pts = 0; //  添加地图点结束
    m_mutex_m_box_recent_hitted->lock();
    m_voxels_recent_visited = voxels_recent_visited ;   // 当前scan的点落在的box
    m_mutex_m_box_recent_hitted->unlock();
    return (m_voxels_recent_visited.size() -  number_of_voxels_before_add);  // number_of_voxels_before_add =0， 返回这个scan落在了多少box
}

// 没有用
void Global_map::render_pts_in_voxels(std::shared_ptr<Image_frame> &img_ptr, std::vector<std::shared_ptr<RGB_pts>> &pts_for_render, double obs_time)
{
    Common_tools::Timer tim;
    tim.tic();
    double u, v;
    int hit_count = 0;
    int pt_size = pts_for_render.size();
    m_last_updated_frame_idx = img_ptr->m_frame_idx;
    for (int i = 0; i < pt_size; i++)
    {

        vec_3 pt_w = pts_for_render[i]->get_pos();
        bool res = img_ptr->project_3d_point_in_this_img(pt_w, u, v, nullptr, 1.0);
        if (res == false)
        {
            continue;
        }
        vec_3 pt_cam = (pt_w - img_ptr->m_pose_w2c_t);
        hit_count++;
        vec_2 gama_bak = img_ptr->m_gama_para;
        img_ptr->m_gama_para = vec_2(1.0, 0.0); // Render using normal value?
        double gray = img_ptr->get_grey_color(u, v, 0);
        vec_3 rgb_color = img_ptr->get_rgb(u, v, 0);
        pts_for_render[i]->update_gray(gray, pt_cam.norm());
        pts_for_render[i]->update_rgb(rgb_color, pt_cam.norm(), vec_3(image_obs_cov, image_obs_cov, image_obs_cov), obs_time);
        img_ptr->m_gama_para = gama_bak;
        // m_rgb_pts_vec[i]->update_rgb( vec_3(gray, gray, gray) );
    }
    // cout << "Render cost time = " << tim.toc() << endl;
    // cout << "Total hit count = " << hit_count << endl;
}

Common_tools::Cost_time_logger cost_time_logger_render("/home/ziv/temp/render_thr.log");

std::atomic<long> render_pts_count ;  // 总共渲染的点的数量
/**
 * @brief 
 * 
 * @param pt_start 开始的box的idx（voxels_for_render的索引）
 * @param pt_end   结束的box的idx
 * @param img_ptr  当前图像      
 * @param voxels_for_render 当前scan访问到的box，即需要渲染的（点云体素）box
 * @param obs_time  当前图像的时间
 * @return double 
 */
static inline double thread_render_pts_in_voxel(const int & pt_start, const int & pt_end, const std::shared_ptr<Image_frame> & img_ptr,
                                                const std::vector<RGB_voxel_ptr> * voxels_for_render, const double obs_time)
{
    vec_3 pt_w; // 这个点在世界坐标系下的xyz
    vec_3 rgb_color;  // 这个点的RGB
    double u, v;  // 世界坐标点在图像中的像素坐标
    double pt_cam_norm;    // 相机指向点的向量
    Common_tools::Timer tim;
    tim.tic();
    // 开始遍历这个线程负责的box
    for (int voxel_idx = pt_start; voxel_idx < pt_end; voxel_idx++)
    {
        // continue;
        RGB_voxel_ptr voxel_ptr = (*voxels_for_render)[ voxel_idx ]; // 取出box指针
        
        // 遍历这个box中的点
        for ( int pt_idx = 0; pt_idx < voxel_ptr->m_pts_in_grid.size(); pt_idx++ )
        {
            pt_w = voxel_ptr->m_pts_in_grid[pt_idx]->get_pos(); // 获得这个点在世界坐标系下的xyz

            // 将3D点投影到图像内染色
            if ( img_ptr->project_3d_point_in_this_img( pt_w, u, v, nullptr, 1.0 ) == false ) // 点不在图像内或者深度值太小，则跳过
            {
                continue;
            }
            // 相机到点的距离
            pt_cam_norm = ( pt_w - img_ptr->m_pose_w2c_t ).norm();
            // double gray = img_ptr->get_grey_color(u, v, 0);
            // pts_for_render[i]->update_gray(gray, pt_cam_norm);
       
            rgb_color = img_ptr->get_rgb( u, v, 0 );     // 这个点的RGB颜色
            if (  voxel_ptr->m_pts_in_grid[pt_idx]->update_rgb(
                     rgb_color, pt_cam_norm, vec_3( image_obs_cov, image_obs_cov, image_obs_cov ), obs_time ) )
            {
                render_pts_count++;// 总共渲染的点的数量++
            }
        }
    }
    double cost_time = tim.toc() * 100;
    return cost_time;
}


std::vector<RGB_voxel_ptr>  g_voxel_for_render;  // 用来存储当前需要渲染的box，等于_voxels_for_render。

/**
 * @brief 点云渲染线程
 * 
 * @param img_ptr             当前图像
 * @param _voxels_for_render  当前scan访问到的box，即需要渲染的（点云体素）box
 * @param obs_time            当前图像的时间
 */
void render_pts_in_voxels_mp(std::shared_ptr<Image_frame> &img_ptr, std::unordered_set<RGB_voxel_ptr> * _voxels_for_render,  const double & obs_time)
{
    Common_tools::Timer tim;
    g_voxel_for_render.clear();
    // 遍历box
    for(Voxel_set_iterator it = (*_voxels_for_render).begin(); it != (*_voxels_for_render).end(); it++)
    {
        g_voxel_for_render.push_back(*it);
    }
    std::vector<std::future<double>> results;
    tim.tic("Render_mp");
    int numbers_of_voxels = g_voxel_for_render.size(); // 需要渲染的box数量
    g_cost_time_logger.record("Pts_num", numbers_of_voxels);
    render_pts_count= 0 ;
    if(USING_OPENCV_TBB)
    {
        // opencv的并行计算库，会并行计算thread_render_pts_in_voxel函数numbers_of_voxels次
        cv::parallel_for_(cv::Range(0, numbers_of_voxels), [&](const cv::Range &r)
                          { thread_render_pts_in_voxel(r.start, r.end, img_ptr, &g_voxel_for_render, obs_time); });
    }
    else{// 作者自己实现的并行计算
        int num_of_threads = std::min(8*2, (int)numbers_of_voxels);
        // results.clear();
        results.resize(num_of_threads);
        tim.tic("Com");
        for (int thr = 0; thr < num_of_threads; thr++)
        {
            // cv::Range range(thr * pt_size / num_of_threads, (thr + 1) * pt_size / num_of_threads);
            int start = thr * numbers_of_voxels / num_of_threads;
            int end = (thr + 1) * numbers_of_voxels / num_of_threads;
            results[thr] = m_thread_pool_ptr->commit_task(thread_render_pts_in_voxel, start, end,  img_ptr, &g_voxel_for_render, obs_time);
        }
        g_cost_time_logger.record(tim, "Com");
        tim.tic("wait_Opm");
        for (int thr = 0; thr < num_of_threads; thr++)
        {
            double cost_time = results[thr].get();
            cost_time_logger_render.record(std::string("T_").append(std::to_string(thr)), cost_time );
        }
        g_cost_time_logger.record(tim, "wait_Opm");
        cost_time_logger_render.record(tim, "wait_Opm");
    }
    // img_ptr->release_image();
    cost_time_logger_render.flush_d();
    g_cost_time_logger.record(tim, "Render_mp");
    g_cost_time_logger.record("Pts_num_r", render_pts_count);
    
}

// 没有用
void Global_map::render_with_a_image(std::shared_ptr<Image_frame> &img_ptr, int if_select)
{

    std::vector<std::shared_ptr<RGB_pts>> pts_for_render;
    // pts_for_render = m_rgb_pts_vec;
    if (if_select)
    {
        selection_points_for_projection(img_ptr, &pts_for_render, nullptr, 1.0);
    }
    else
    {
        pts_for_render = m_rgb_pts_vec;
    }
    render_pts_in_voxels(img_ptr, pts_for_render);
}

/**
 * @brief 将上个scan的点投影到当前相机坐标下，去掉投影过程中的重复点，遮挡点
 *
 *     获取当前图像帧可以用于LK光流跟踪的像素点，和其对应的空间点
 *     （1）获取可以用来投影到图像上的空间点，获取最近被访问过的空间点，即最新增加至全局地图中的空间点，这些点大概率是图像和LIDAR同时看到的点，
 *     因此可以投影至图像上得到像素点。
 *     （2）将空间点投影到图像上，获取图二维点，这些点在后面用来进行光流追踪；
 * 
 * @param image_pose 当前相机帧位姿
 * @param pc_out_vec    上个scan通过相机投影去掉重复点后的点  
 * @param pc_2d_out_vec 上个scan通过相机投影到当前帧中的像素坐标
 * @param minimum_dis   像素间的最小距离，小于这个距离当作同一个像素
 * @param skip_step     降采样率
 * @param use_all_pts   default = 0
 */
void Global_map::selection_points_for_projection(std::shared_ptr<Image_frame> &image_pose, std::vector<std::shared_ptr<RGB_pts>> *pc_out_vec,
                                                            std::vector<cv::Point2f> *pc_2d_out_vec, double minimum_dis,
                                                            int skip_step,
                                                            int use_all_pts)
{
    Common_tools::Timer tim;
    tim.tic();

    ///清空一下，输出点云
    if (pc_out_vec != nullptr)
    {
        pc_out_vec->clear();
    }

    ///清空一下点云在图像上的投影
    if (pc_2d_out_vec != nullptr)
    {
        pc_2d_out_vec->clear();
    }

    ///一个存索引。一个存深度
    Hash_map_2d<int, int> mask_index;  // 一个scan的hashmap，防止重复添加
    Hash_map_2d<int, float> mask_depth;

    std::map<int, cv::Point2f> map_idx_draw_center;  // first -> 特征点索引  second-》 特征点坐标（栅格离散坐标）
    std::map<int, cv::Point2f> map_idx_draw_center_raw_pose;  // first -> 特征点索引  second-》 特征点坐标

    int u, v;
    double u_f, v_f;
    // cv::namedWindow("Mask", cv::WINDOW_FREERATIO);
    int acc = 0;    // 接受点
    int blk_rej = 0; // 拒绝点
    // int pts_size = m_rgb_pts_vec.size();
    std::vector<std::shared_ptr<RGB_pts>> pts_for_projection; // 需要投影的点（上一帧scan的点）
    m_mutex_m_box_recent_hitted->lock();

    ///获取最近被访问过的体素，即最新增加至全局地图中的体素，
    ///这些点大概率是图像和LIDAR同时看到的点，因此可以投影至图像上得到像素点。
    std::unordered_set< std::shared_ptr< RGB_Voxel > > boxes_recent_hitted = m_voxels_recent_visited; // 上一帧scan落在的box集合
    m_mutex_m_box_recent_hitted->unlock();

    ///把最近访问过的体素中的地图点提取出来，准备投影
    if ( (!use_all_pts) && boxes_recent_hitted.size())  // 每个box只取一个点
    {
        m_mutex_rgb_pts_in_recent_hitted_boxes->lock();
        //对体素单元进行遍历
        for(Voxel_set_iterator it = boxes_recent_hitted.begin(); it != boxes_recent_hitted.end(); it++)
        {
            // pts_for_projection.push_back( (*it)->m_pts_in_grid.back() );

            //如果体素内有地图点，则添加一个地图点进去，有多个地图点也添加一个进去
            if ( ( *it )->m_pts_in_grid.size() ) // 每个box只取一个点
            {
                 pts_for_projection.push_back( (*it)->m_pts_in_grid.back() );
                // pts_for_projection.push_back( ( *it )->m_pts_in_grid[ 0 ] );
                // pts_for_projection.push_back( ( *it )->m_pts_in_grid[ ( *it )->m_pts_in_grid.size()-1 ] );
            }
        }

        m_mutex_rgb_pts_in_recent_hitted_boxes->unlock();
    }
    else  // 用box中全部的点
    {
        pts_for_projection = m_rgb_pts_vec;
    }

    //将空间点投影到图像上，获取图像二维点，这些点在后面用来进行光流跟踪；
    int pts_size = pts_for_projection.size();

    // 遍历每一个需要投影的点
    for (int pt_idx = 0; pt_idx < pts_size; pt_idx += skip_step)
    {

        //地图点的坐标取出来
        vec_3 pt = pts_for_projection[pt_idx]->get_pos();

        //地图点到相机的距离需要满足最大与最小观测阈值
        double depth = (pt - image_pose->m_pose_w2c_t).norm();
        if (depth > m_maximum_depth_for_projection)
        {
            continue;
        }
        if (depth < m_minimum_depth_for_projection)
        {
            continue;
        }

        // 地图点根据当前帧的位姿投影到图像坐标系下，并进行深度值校验
        bool res = image_pose->project_3d_point_in_this_img(pt, u_f, v_f, nullptr, 1.0); // 将3D点投影到图像内染色
        if (res == false)
        {
            continue;
        }

        //像素坐标取整，最小像素值为50，防止取到边缘处的点
        u = std::round(u_f / minimum_dis) * minimum_dis; // 作者Why can not work  I：应该用int
        v = std::round(v_f / minimum_dis) * minimum_dis;
        // 不存在 || 存在但之前存的深度大于当前点
        if ((!mask_depth.if_exist(u, v)) || mask_depth.m_map_2d_hash_map[u][v] > depth)
        {
            acc++;
            if (mask_index.if_exist(u, v)) //如果已经存在，把旧的点删掉（相当于只存更近的点）
            {
                // erase old point
                int old_idx = mask_index.m_map_2d_hash_map[u][v];
                blk_rej++;
                map_idx_draw_center.erase(map_idx_draw_center.find(old_idx));
                map_idx_draw_center_raw_pose.erase(map_idx_draw_center_raw_pose.find(old_idx));
            }
            // 赋新值  存储新的索引和对应的深度值
            mask_index.m_map_2d_hash_map[u][v] = (int)pt_idx;
            mask_depth.m_map_2d_hash_map[u][v] = (float)depth;
            map_idx_draw_center[pt_idx] = cv::Point2f(v, u);
            map_idx_draw_center_raw_pose[pt_idx] = cv::Point2f(u_f, v_f);
        }
    }

    if (pc_out_vec != nullptr)
    {
        //把符合条件的点丢进去
        for (auto it = map_idx_draw_center.begin(); it != map_idx_draw_center.end(); it++)
        {
            // pc_out_vec->push_back(m_rgb_pts_vec[it->first]);
            pc_out_vec->push_back(pts_for_projection[it->first]);
        }
    }

    //防止指针为空
    if (pc_2d_out_vec != nullptr)
    {
        //把符合条件的点丢进去
        for (auto it = map_idx_draw_center.begin(); it != map_idx_draw_center.end(); it++)
        {
            pc_2d_out_vec->push_back(map_idx_draw_center_raw_pose[it->first]);
        }
    }

}

void Global_map::save_to_pcd(std::string dir_name, std::string _file_name, int save_pts_with_views )
{
    Common_tools::Timer tim;
    Common_tools::create_dir(dir_name);
    std::string file_name = std::string(dir_name).append(_file_name);
    scope_color(ANSI_COLOR_BLUE_BOLD);
    cout << "Save Rgb points to " << file_name << endl;
    fflush(stdout);
    pcl::PointCloud<pcl::PointXYZRGB> pc_rgb;
    long pt_size = m_rgb_pts_vec.size();
    pc_rgb.resize(pt_size);
    long pt_count = 0;
    for (long i = pt_size - 1; i > 0; i--)
    //for (int i = 0; i  <  pt_size; i++)
    {
        if ( i % 1000 == 0)
        {
            cout << ANSI_DELETE_CURRENT_LINE << "Saving offline map " << (int)( (pt_size- 1 -i ) * 100.0 / (pt_size-1) ) << " % ...";
            fflush(stdout);
        }

        if (m_rgb_pts_vec[i]->m_N_rgb < save_pts_with_views)
        {
            continue;
        }
        pcl::PointXYZRGB pt;
        pc_rgb.points[ pt_count ].x = m_rgb_pts_vec[ i ]->m_pos[ 0 ];
        pc_rgb.points[ pt_count ].y = m_rgb_pts_vec[ i ]->m_pos[ 1 ];
        pc_rgb.points[ pt_count ].z = m_rgb_pts_vec[ i ]->m_pos[ 2 ];
        pc_rgb.points[ pt_count ].r = m_rgb_pts_vec[ i ]->m_rgb[ 2 ];
        pc_rgb.points[ pt_count ].g = m_rgb_pts_vec[ i ]->m_rgb[ 1 ];
        pc_rgb.points[ pt_count ].b = m_rgb_pts_vec[ i ]->m_rgb[ 0 ];
        pt_count++;
    }
    cout << ANSI_DELETE_CURRENT_LINE  << "Saving offline map 100% ..." << endl;
    pc_rgb.resize(pt_count);
    cout << "Total have " << pt_count << " points." << endl;
    tim.tic();
    cout << "Now write to: " << file_name << endl; 
    pcl::io::savePCDFileBinary(std::string(file_name).append(".pcd"), pc_rgb);
    cout << "Save PCD cost time = " << tim.toc() << endl;
}

void Global_map::save_and_display_pointcloud(std::string dir_name, std::string file_name, int save_pts_with_views)
{
    save_to_pcd(dir_name, file_name, save_pts_with_views);
    scope_color(ANSI_COLOR_WHITE_BOLD);
    cout << "========================================================" << endl;
    cout << "Open pcl_viewer to display point cloud, close the viewer's window to continue mapping process ^_^" << endl;
    cout << "========================================================" << endl;
    system(std::string("pcl_viewer ").append(dir_name).append("/rgb_pt.pcd").c_str());
}
