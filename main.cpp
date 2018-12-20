#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <cassert>

#include <gtsam/geometry/Point2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h> // 一元因子，系统先验
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/GeneralSFMFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/Values.h>

using namespace std;
using namespace cv;

const int IMAGE_DOWNSAMPLE = 4; // 对图像进行下采样以加快处理速度
const double FOCAL_LENGTH = 4308 / IMAGE_DOWNSAMPLE;   //下采样后，根据JPEG EXIF数据猜测焦距（以像素为单位）
const int MIN_LANDMARK_SEEN = 3;    // 必须看到使用3D点（界标）的相机视图的最小数量

const std::vector<std::string> IMAGES = {
        "/home/jxq/SFM-github/desk/DSC02638.JPG",
        "/home/jxq/SFM-github/desk/DSC02639.JPG",
        "/home/jxq/SFM-github/desk/DSC02640.JPG",
        "/home/jxq/SFM-github/desk/DSC02641.JPG",
        "/home/jxq/SFM-github/desk/DSC02642.JPG"
//    "/home/jxq/CLionProjects/test/1217_6/1217_6.jpg",
//    "/home/jxq/CLionProjects/test/1217_6/1217_7.jpg",
//    "/home/jxq/CLionProjects/test/1217_6/1217_8.jpg",
//    "/home/jxq/CLionProjects/test/1217_6/1217_9.jpg",
//    "/home/jxq/CLionProjects/test/1217_6/1217_10.jpg"
};

struct SFM_Helper
{
    struct ImagePose
    {
        Mat img;    // 用于显示的下采样图像
        Mat desc;   // 特征描述符
        vector<KeyPoint> kp;    // 关键点

        Mat T;  // 4x4 位置变换矩阵
        Mat P;  // 3x4 投影矩阵

        // 别名以澄清下面的用法
        using kp_idx_t = size_t;
        using landmark_idx_t = size_t;
        using img_idx_t = size_t;

        map<kp_idx_t, map<img_idx_t, kp_idx_t>> kp_matches; // 在其他图像中的关键点匹配
        map<kp_idx_t, landmark_idx_t> kp_landmark;  // seypoint to 3d point

        // helper
        kp_idx_t& kp_match_idx( size_t kp_idx, size_t img_idx )
        {
            return kp_matches[kp_idx][img_idx];
        }
        bool kp_match_exist( size_t kp_idx, size_t img_idx )
        {
            return kp_matches[kp_idx].count(img_idx) > 0;
        }

        landmark_idx_t& kp_3d( size_t kp_idx )
        {
            return kp_landmark[kp_idx];
        }
        bool kp_3d_exist( size_t kp_idx )
        {
            return kp_landmark.count(kp_idx) > 0;
        }

    };

    // 3D point
    struct Landmark
    {
        Point3f pt;
        int seen = 0;   // 有多少摄像机看到了这一点
    };

    vector<ImagePose> img_pose;
    vector<Landmark> landmark;
};

int main( int argc, char** argv)
{
    SFM_Helper SFM;

    // 查找匹配特征
    {
        Ptr<AKAZE> feature = AKAZE::create();
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

        namedWindow("img", WINDOW_NORMAL);
        // int height = 3648;
        // int weight = 5472;
        // 提取特征
        for ( auto f : IMAGES )
        {
            SFM_Helper::ImagePose a;

            Mat img = imread(f);
            assert( !img.empty() );
            // resize(img, img, Size(weight, height) );
            // cout << img.size() << endl;
            resize( img, img, img.size()/IMAGE_DOWNSAMPLE );    // 下采样
            a.img = img;
            cvtColor( img, img, COLOR_BGR2GRAY);    // 变为灰度图像

            feature->detect( img, a.kp );
            feature->compute( img, a.kp, a.desc );

            SFM.img_pose.emplace_back(a);
        }

        // 在所有图像中间匹配特征
        for ( size_t i = 0; i < SFM.img_pose.size() - 1; i++ )
        {
            auto &img_pose_i = SFM.img_pose[i];
            for ( size_t j = i+1; j < SFM.img_pose.size(); j++ )
            {
                auto &img_pose_j = SFM.img_pose[j];
                vector<vector<DMatch>> matches;
                vector<Point2f> src, dst;   // 存储请求描述子和训练描述子对应的关键点坐标
                vector<uchar> mask;
                vector<int> i_kp, j_kp;

                // 2 最近邻匹配
                matcher->knnMatch(img_pose_i.desc, img_pose_j.desc, matches, 2);

                for ( auto &m : matches )
                {
                    // cout << m[0].distance << "\t" << m[1].distance << endl;
                    if ( m[0].distance < 0.7*m[1].distance )    // 当最优匹配距离远小于次优匹配时，存储最优匹配相关参数
                    {
                        src.push_back(img_pose_i.kp[m[0].queryIdx].pt);
                        dst.push_back(img_pose_j.kp[m[0].trainIdx].pt);

                        i_kp.push_back(m[0].queryIdx);
                        j_kp.push_back(m[0].trainIdx);
                    }
                }

                // cout << src.size() << " " << dst.size() << endl;
                // 使用基本矩阵约束筛选不匹配项 之后mask中存放的是剩余的感兴趣点
                findFundamentalMat(src, dst, FM_RANSAC, 3.0, 0.99, mask);


                Mat canvas = img_pose_i.img.clone();
                canvas.push_back(img_pose_j.img.clone());   // 将两张图片合并成一张图片

                for ( size_t k = 0; k < mask.size(); k++ )
                {
                    if ( mask[k] )  // 感兴趣的部分掩膜值为1
                    {
                        img_pose_i.kp_match_idx(i_kp[k], j) = j_kp[k];
                        img_pose_j.kp_match_idx(j_kp[k], i) = i_kp[k];

                        // 对应关键点连线
                        line(canvas, src[k], dst[k] + Point2f(0, img_pose_i.img.rows), Scalar(0, 0, 255), 2);
                    }
                }

                int good_matches = sum(mask)[0];    // 计算arr各通道所有像素总和
                // cout << sum(mask) << endl;       // [1167, 0, 0, 0]
                assert( good_matches >= 10 );

                cout << "Feature matching " << i << " " << j << " ==> " << good_matches << "/" << matches.size() << endl;

                resize(canvas, canvas, canvas.size() / 2);
                imshow("img", canvas);
                waitKey(1);
            }
        }
    }

    // 恢复当前图像和三角形点之间的运动
    {
        // 设置摄像机矩阵
        double cx = SFM.img_pose[0].img.size().width / 2;
        double cy = SFM.img_pose[0].img.size().height / 2;

        Point2d pp(cx, cy); // 中心点位置

        Mat K = Mat::eye(3, 3, CV_64F);

        K.at<double>(0, 0) = FOCAL_LENGTH;
        K.at<double>(1, 1) = FOCAL_LENGTH;
        K.at<double>(0, 2) = cx;
        K.at<double>(1, 2) = cy;

        cout << endl << "initial camera matrix K " << endl << K << endl << endl;

        // 设置第一幅图片的旋转矩阵和投影矩阵 即设置世界坐标系中参数
        SFM.img_pose[0].T = Mat::eye(4, 4, CV_64F);
        SFM.img_pose[0].P = K * Mat::eye(3, 4, CV_64F);

        // cout << SFM.img_pose[0].T << endl;
        // cout << SFM.img_pose[0].P << endl;

        for ( size_t i = 0; i < SFM.img_pose.size()-1; i++ )
        {
            auto &prev = SFM.img_pose[i];
            auto &cur = SFM.img_pose[i+1];

            vector<Point2f> src, dst;
            vector<size_t> kp_used;

            for ( size_t k = 0; k < prev.kp.size(); k++ )
            {
                if ( prev.kp_match_exist(k, i+1) )  // 如果之前一张图片中的关键点在当前图片中存在匹配
                {
                    size_t match_idx = prev.kp_match_idx(k, i+1);   // 得到在前图像在当前图像的关键点索引

                    src.push_back( prev.kp[k].pt );         // 之前图像中关键点索引
                    dst.push_back( cur.kp[match_idx].pt );  // 当前图像中关键点索引

                    kp_used.push_back(k);   // 存储相邻两张图片存在对应的关键的索引（当前图片的索引）
                }
            }

            Mat mask;

            // cout << dst.size() << endl;

            // NOTE: pose from dst to src
            Mat E = findEssentialMat( dst, src, FOCAL_LENGTH, pp, RANSAC, 0.999, 1.0, mask ); // 本质矩阵
            Mat local_R, local_t;

            // 恢复位姿 src 相对于 dst 的 旋转和位移
            recoverPose( E, dst, src, local_R, local_t, FOCAL_LENGTH, pp, mask );

            // 局部变换
            Mat T = Mat::eye(4, 4, CV_64F);
            local_R.copyTo(T(Range(0, 3), Range(0, 3)));    // 行 列
            local_t.copyTo(T(Range(0, 3), Range(3, 4)));

            // cout << local_R << endl;
            // cout << local_t << endl;
            // cout << T << endl;


            // 累加变换 相对于世界坐标系（前图片坐标系）的外部参数矩阵
            cur.T = prev.T * T;

            // 投影矩阵
            Mat R = cur.T(Range(0, 3), Range(0, 3));
            Mat t = cur.T(Range(0, 3), Range(3, 4));

            Mat P(3, 4, CV_64F);



            P(Range(0, 3), Range(0, 3)) = R.t();
            P(Range(0, 3), Range(3, 4)) = -R.t() * t;
            P = K * P;  // P= K[R|−RC] = K[R|T]    t = -RC

            cur.P = P;

            Mat points4D;   // 齐次坐标中重建点的4×N阵列。
            triangulatePoints(prev.P, cur.P, src, dst, points4D);

            // cout << points4D.cols << endl;

            // 将新的3D点缩放为与现有3D点相似（界标）
            // 配对3D点之间距离的使用率
            if ( i > 0 )
            {
                double scale = 0;
                int count = 0;

                Point3f prev_camera;    // 前一副图像相机位置

                prev_camera.x = prev.T.at<double>(0, 3);
                prev_camera.y = prev.T.at<double>(1, 3);
                prev_camera.z = prev.T.at<double>(2, 3);

                vector<Point3f> new_pts;
                vector<Point3f> existing_pts;

                // imshow("mask", mask);

                for ( size_t j = 0; j < kp_used.size(); j++ )
                {
                    size_t k = kp_used[j];
                    if ( mask.at<uchar>(j) && prev.kp_match_exist(k, i+1) && prev.kp_3d_exist(k) )
                    {
                        Point3f pt3d;

                        pt3d.x = points4D.at<float>(0, j) / points4D.at<float>(3, j);
                        pt3d.y = points4D.at<float>(1, j) / points4D.at<float>(3, j);
                        pt3d.z = points4D.at<float>(2, j) / points4D.at<float>(3, j);

                        size_t idx = prev.kp_3d(k);
                        Point3f avg_landmark = SFM.landmark[idx].pt / (SFM.landmark[idx].seen - 1);

                        new_pts.push_back(pt3d);
                        existing_pts.push_back(avg_landmark);
                    }
                }
                // cout << new_pts.size() << endl;

                // 所有可能的点配对的距离比率
                for ( size_t j = 0; j < new_pts.size()-1; j++ )
                {
                    for ( size_t k = j+1; k < new_pts.size(); k++ )
                    {
                        double s = norm(existing_pts[j] - existing_pts[k]) / norm(new_pts[j] - new_pts[k]);

                        scale += s;
                        count++;
                    }
                }

                assert(count > 0);
                scale /= count;

                cout << "image " << (i+1) << " ==> " << i << " scale=" << scale << " count=" << count << endl;

                // 应用标度，重新计算T矩阵和P矩阵
                local_t *= scale;

                // 局部旋转
                Mat T = Mat::eye(4, 4, CV_64F);
                local_R.copyTo(T(Range(0, 3), Range(0, 3)));
                local_t.copyTo(T(Range(0, 3), Range(3, 4)));

                // 累加旋转
                cur.T = prev.T * T;

                // 投影矩阵
                R = cur.T(Range(0, 3), Range(0, 3));
                t = cur.T(Range(0, 3), Range(3, 4));

                Mat P(3, 4, CV_64F);
                P(Range(0, 3), Range(0, 3)) = R.t();
                P(Range(0, 3), Range(0, 4)) = -R.t() * t;
                P = K * P;

                cur.P = P;

                triangulatePoints(prev.P, cur.P, src, dst, points4D);
            }

            // 找到好的三角形点
            for ( size_t j = 0; j < kp_used.size(); j++ )
            {
                if ( mask.at<uchar>(j) )
                {
                    size_t k = kp_used[j];
                    size_t match_idx = prev.kp_match_idx(k, i+1);   // 前一副图像k关键点在当前图像关键点索引

                    Point3f pt3d;

                    pt3d.x = points4D.at<float>(0, j) / points4D.at<float>(3, j);
                    pt3d.y = points4D.at<float>(1, j) / points4D.at<float>(3, j);
                    pt3d.z = points4D.at<float>(2, j) / points4D.at<float>(3, j);

                    if (prev.kp_3d_exist(k))
                    {
                        // 找到与现有界标匹配的项
                        cur.kp_3d(match_idx) = prev.kp_3d(k);

                        SFM.landmark[prev.kp_3d(k)].pt += pt3d;
                        SFM.landmark[cur.kp_3d(match_idx)].seen++;
                    }
                    else
                    {
                        // 添加新的3d点
                        SFM_Helper::Landmark landmark;

                        landmark.pt = pt3d;
                        landmark.seen = 2;

                        SFM.landmark.push_back(landmark);

                        prev.kp_3d(k) = SFM.landmark.size() - 1;
                        cur.kp_3d(match_idx)= SFM.landmark.size() - 1;
                    }
                }
            }
        }

        // 平均标志性三维位置
        for (auto &l : SFM.landmark)
        {
            if (l.seen >= 3)
            {
                l.pt /= (l.seen - 1);
            }
        }
    }

    // GTSAM 光束法调差
    gtsam::Values result;
    {
        using namespace gtsam;

        double cx = SFM.img_pose[0].img.size().width / 2;
        double cy = SFM.img_pose[0].img.size().height / 2;

        Cal3_S2 K(FOCAL_LENGTH, FOCAL_LENGTH, 0, cx, cy);   // 内参数矩阵
        // 噪声定义 对角线矩阵：
        noiseModel::Isotropic::shared_ptr measurement_noise = noiseModel::Isotropic::Sigma(2, 2.0); // pixel error in (x,y)

        // 定义因子图
        NonlinearFactorGraph graph;
        Values initial; // 定义初始值容器


        // Poses
        for (size_t i = 0; i < SFM.img_pose.size(); i++)
        {
            auto &img_pose = SFM.img_pose[i];

            // 旋转矩阵
            Rot3 R(
                    img_pose.T.at<double>(0, 0),
                    img_pose.T.at<double>(0, 1),
                    img_pose.T.at<double>(0, 2),

                    img_pose.T.at<double>(1, 0),
                    img_pose.T.at<double>(1, 1),
                    img_pose.T.at<double>(1, 2),

                    img_pose.T.at<double>(2, 0),
                    img_pose.T.at<double>(2, 1),
                    img_pose.T.at<double>(2, 2)
                    );

            Point3 *t = new Point3(img_pose.T.at<double>(0, 3), img_pose.T.at<double>(1,3), img_pose.T.at<double>(2,3));

            // A 3D pose (R,t) : (Rot3,Point3) 位姿
            Pose3 pose(R, *t);

            // 为第一个图像添加前缀 生成factor
            if (i == 0)
            {
                // 噪声    Vector3三维向量：表示3D的向量和点。包含位置、方向（朝向）、欧拉角的信息
                noiseModel::Diagonal::shared_ptr pose_noise = noiseModel::Diagonal::Sigmas(
                        (Vector(6) << Vector3::Constant(0.1), Vector3::Constant(0.1)).finished()
                        );              // 常值矩阵的一个表达式
                graph.emplace_shared<PriorFactor<Pose3>>(Symbol('x', 0), pose, pose_noise); // 添加到图中
            }

            initial.insert(Symbol('x', i), pose);   // 加入一个变量 arg1：变量的标签 arg2:这个变量的值

            // 地标SEEN
            for (size_t k = 0; k < img_pose.kp.size(); k++)
            {
                if (img_pose.kp_3d_exist(k))
                {
                    size_t landmark_id = img_pose.kp_3d(k);

                    if (SFM.landmark[landmark_id].seen >= MIN_LANDMARK_SEEN)
                    {
                        Point2 *pt = new Point2(img_pose.kp[k].pt.x, img_pose.kp[k].pt.y);

                        graph.emplace_shared<GeneralSFMFactor2<Cal3_S2>>(*pt,
                                measurement_noise, Symbol('x', i), Symbol('l', landmark_id), Symbol('K', 0));
                    }
                }
            }
        }


        // 在校准上添加优先级。
        initial.insert(Symbol('K', 0), K);

        noiseModel::Diagonal::shared_ptr cal_noise = noiseModel::Diagonal::Sigmas((Vector(5) << 100, 100, 0.01 /*skew*/, 100, 100).finished());
        graph.emplace_shared<PriorFactor<Cal3_S2>>(Symbol('K', 0), K, cal_noise);


        // 初始化界标估计
        bool init_prior = false;

        for (size_t i = 0; i < SFM.landmark.size(); i++)
        {
            if (SFM.landmark[i].seen >= MIN_LANDMARK_SEEN)
            {
                Point3f &p = SFM.landmark[i].pt;

                initial.insert(Symbol('l', i), Point3(p.x, p.y, p.z));

                if (!init_prior)
                {
                    init_prior = true;
                    noiseModel::Isotropic::shared_ptr point_noise = noiseModel::Isotropic::Sigma(3, 0.1);
                    Point3 p(SFM.landmark[i].pt.x, SFM.landmark[i].pt.y, SFM.landmark[i].pt.z);
                    graph.emplace_shared<PriorFactor<Point3>>(Symbol('l', i), p, point_noise);
                }
            }
        }


        result = LevenbergMarquardtOptimizer(graph, initial).optimize();

        cout << endl;
        cout << "initial graph error = " << graph.error(initial) << endl;
        cout << "final graph error = " << graph.error(result) << endl;

    }


    return 0;
}








































