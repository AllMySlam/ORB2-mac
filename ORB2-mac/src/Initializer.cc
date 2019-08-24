/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Initializer.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include "Optimizer.h"
#include "ORBmatcher.h"

#include<thread>

namespace ORB_SLAM2
{

Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations)
{
    mK = ReferenceFrame.mK.clone();

    mvKeys1 = ReferenceFrame.mvKeysUn;

    mSigma = sigma;
    mSigma2 = sigma*sigma;
    mMaxIterations = iterations;
}

/**
 * @brief 类初始化函数
 * 并行地计算基础矩阵和单应性矩阵，选取其中一个模型，恢复出最开始两帧之间的相对姿态以及点云
 * @param CurrentFrame      当前帧和第一帧参考帧匹配，三角变换得到3D点
 * @param vMatches12        当前帧 特征点的匹配信息
 * @param R21               旋转矩阵
 * @param t21               平移矩阵
 * @param vP3D              恢复出的3D点
 * @param vbTriangulated    符合三角变换 的 3D点
 */
bool Initializer::Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                             vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
{
    // Fill structures with current keypoints and matches with reference frame
    // Reference Frame: 1, Current Frame: 2
    // Frame2 当前帧畸变校正后的关键点
    mvKeys2 = CurrentFrame.mvKeysUn;// 当前帧(2) 关键点

    mvMatches12.clear();// mvMatches12记录匹配上的特征点对
    mvMatches12.reserve(mvKeys2.size());
    // mvbMatched1记录每个特征点是否有匹配的特征点，
    // 这个变量后面没有用到，后面只关心匹配上的特征点
    mvbMatched1.resize(mvKeys1.size());//参考帧的特征点mvKeys1大小

// 步骤1：根据 matcher.SearchForInitialization 得到的初始匹配点对，筛选后得到好的特征匹配点对
// 步骤1：组织特征点对
    for(size_t i=0, iend=vMatches12.size();i<iend; i++)
    {
        if(vMatches12[i]>=0)// 帧2特征点 有匹配
        {
            mvMatches12.push_back(make_pair(i,vMatches12[i]));
            mvbMatched1[i]=true;
        }
        else
            mvbMatched1[i]=false;
    }

    const int N = mvMatches12.size();// 匹配上的特征点的个数

    // Indices for minimum set selection
    // 新建一个容器vAllIndices，生成0到N-1的数作为特征点的索引
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }

    // Generate sets of 8 points for each RANSAC iteration
// 步骤2： 在所有匹配特征点对中随机选择8对特征匹配点对为一组，共选择mMaxIterations组
        // 用于FindHomography和FindFundamental求解
        // mMaxIterations:200
        // 随机采样序列 最大迭代次数 随机序列 8点法
    mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));

    DUtils::Random::SeedRandOnce(0);

    for(int it=0; it<mMaxIterations; it++)
    {
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            // 产生0到N-1的随机数
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
            // idx表示哪一个索引对应的特征点被选中
            int idx = vAvailableIndices[randi];

            mvSets[it][j] = idx;

            // randi对应的索引已经被选过了，从容器中删除
            // randi对应的索引用最后一个元素替换，并删掉最后一个元素，这样子效率最高
            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }

    // Launch threads to compute in parallel a fundamental matrix and a homography
// 步骤3：调用多线程分别用于计算fundamental matrix和homography
    vector<bool> vbMatchesInliersH, vbMatchesInliersF;
    float SH, SF;
    cv::Mat H, F;

    // ref是引用的功能:http://en.cppreference.com/w/cpp/utility/functional/ref
    // 计算homograpy并打分
    thread threadH(&Initializer::FindHomography,this,ref(vbMatchesInliersH), ref(SH), ref(H));
    // 计算fundamental matrix并打分
    thread threadF(&Initializer::FindFundamental,this,ref(vbMatchesInliersF), ref(SF), ref(F));

    // Wait until both threads have finished
    threadH.join();
    threadF.join();

// 步骤4：计算得分比例，选取某个模型
    // 从两个模型 H F 得分为 Sh   Sf 中选着一个最优秀的模型的方法为
    // Compute ratio of scores
    float RH = SH/(SH+SF);

// 步骤5：根据评价得分，从单应矩阵H 或 基础矩阵F中恢复R,t
    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
    if(RH>0.40)// 更偏向于平面，使用单应矩阵恢复
        return ReconstructH(vbMatchesInliersH,H,mK,R21,t21,vP3D,vbTriangulated,1.0,50);
    else //if(pF_HF>0.6)// 偏向于非平面，使用基础矩阵恢复
        return ReconstructF(vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,1.0,50);

    return false;
}

/**
 * @brief 计算单应矩阵   随机采样序列 8点  采用归一化的直接线性变换（normalized DLT）
 * 假设场景为平面情况下通过前两帧求取Homography矩阵(current frame 2 到 reference frame 1)
 * 在最大迭代次数内 调用 ComputeH21 计算  使用 CheckHomography 计算单应 得分
 * 并得到该模型的评分
 * 在最大迭代次数内 保留 最高得分的 单应矩阵
 * @param vbMatchesInliers     返回的 符合 变换的 匹配点 内点 标志
 * @param score                变换得分
 * @param H21                  单应矩阵
 */
void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
{
    // Number of putative matches
    const int N = mvMatches12.size();

// 步骤1：
    // 将mvKeys1和mvKey2归一化到均值为0，一阶绝对矩为1，归一化矩阵分别为T1、T2
    //2D-2D点对 求变换矩阵前先进行标准化  去均值点坐标 * 绝对矩倒数
    //标准化矩阵  * 点坐标    =   标准化后的的坐标
    // 点坐标    =    标准化矩阵 逆矩阵 * 标准化后的的坐标
    // Normalize coordinates
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2inv = T2.inv();

    // Best Results variables
    // 最终最佳的MatchesInliers与得分
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat H21i, H12i;
    // 每次RANSAC的MatchesInliers与得分
    vector<bool> vbCurrentInliers(N,false);//内点标志
    float currentScore;

// 步骤2：随机采样序列迭代求解
    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++)
    {
//步骤3：随机8对点对
        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            int idx = mvSets[it][j];

            // vPn1i和vPn2i为匹配的特征点对的坐标
            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }

        /*
        用于还原 单应矩阵 下面需要用到：
        p1'  ------> Hn -------> p2'   , p2'   = Hn*p1'
        T1*p1 -----> Hn -------> T2*p2 , T2*p2 = Hn*(T1*p1)
        左乘 T2逆 ，得到   p2 = T2逆 * Hn*(T1*p1)= H21i*p1
        H21i = T2逆 * Hn * T1
        */
// 步骤4：计算单应矩阵
        cv::Mat Hn = ComputeH21(vPn1i,vPn2i);
        // 恢复原始的均值和尺度
        H21i = T2inv*Hn*T1;
        H12i = H21i.inv();

        // 计算单应转换矩阵得分
        /*
        *  变换矩阵评分方法
        * SM=∑i( ρM( d2cr(xic,xir,M)   +   ρM( d2rc(xic,xir,M ) )
        *  d2cr 为 2D-2D点对通过转换矩阵的对称转换误差
        *
        * ρM 函数为 ρM(d^2)  = 0            当  d^2 > 阈值(单应矩阵时 为 5.99  基础矩阵时为 3.84)
        *                   最高分 - d^2    当  d^2 < 阈值
        *                                   最高分 均为 5.99
        */
// 步骤5：计算单应H的得分，由对应匹配点对的对称转换误差求得
        // 利用重投影误差为当次RANSAC的结果评分
        currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);//mSigma=1.0
// 步骤6：保留最高得分对应的单应，得到最优的vbMatchesInliers与score
        if(currentScore>score)
        {
            H21 = H21i.clone();//保留较高得分的单应
            vbMatchesInliers = vbCurrentInliers;//对应的匹配点对
            score = currentScore;// 最高的得分
        }
    }
}

// 计算基础矩阵   随机采样序列 8点  采用归一化的直接线性变换（normalized DLT）
// 在最大迭代次数内 调用 ComputeH21 计算  使用 CheckHomography 计算单应 得分
// 在最大迭代次数内 保留 最高得分的 单应矩阵
/**
 * @brief 计算基础矩阵
 *
 * 假设场景为非平面情况下通过前两帧求取Fundamental矩阵(current frame 2 到 reference frame 1),并得到该模型的评分
 */
void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
    // Number of putative matches
    const int N = vbMatchesInliers.size();
    /*
     *【1】2D-2D点对 求变换矩阵前先进行标准化  去均值点坐标 * 绝对矩倒数
     * 标准化矩阵  * 点坐标    =   标准化后的的坐标
     * 点坐标    =    标准化矩阵 逆矩阵 * 标准化后的的坐标
     */
    // Normalize coordinates
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2t = T2.t();

    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat F21i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(int j=0; j<8; j++)
        {
            int idx = mvSets[it][j];

            // vPn1i和vPn2i为匹配的特征点对的坐标
            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }
        /*
        再者用于还原 单应矩阵 下面需要用到：
        p1'  ------> Hn -------> p2'   , p2'   = Hn*p1'
        T1*p1 -----> Hn -------> T2*p2 , T2*p2 = Hn*(T1*p1)
        左乘 T2逆 ，得到   p2 = T2逆 * Hn*(T1*p1)= H21i*p1
        H21i = T2逆 * Hn * T1
        */
        cv::Mat Fn = ComputeF21(vPn1i,vPn2i);
        // 恢复原始的均值和尺度
        F21i = T2t*Fn*T1;

        currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

        if(currentScore>score)
        {
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}

// 计算单应矩阵  8对点对 每个点提供两个约束   A × h = 0 求h 奇异值分解 求 h
// // 通过svd进行最小二乘求解
// 参考   http://www.fengbing.net/
// |x'|     | h1 h2 h3 ||x|
// |y'| = a | h4 h5 h6 ||y|  简写: x' = a H x, a为一个尺度因子
// |1 |     | h7 h8 h9 ||1|
// 使用DLT(direct linear tranform)求解该模型
// x' = a H x
// ---> (x') 叉乘 (H x)  = 0
// ---> Ah = 0
// A = | 0  0  0 -x -y -1 xy' yy' y'|  h = | h1 h2 h3 h4 h5 h6 h7 h8 h9 |
//     |-x -y -1  0  0  0 xx' yx' x'|
// 通过SVD求解Ah = 0，A'A最小特征值对应的特征向量即为解
/**
 * @brief 从特征点匹配求homography（normalized DLT）
 *
 * @param  vP1 归一化后的点, in reference frame
 * @param  vP2 归一化后的点, in current frame
 * @return     单应矩阵
 * @see        Multiple View Geometry in Computer Vision - Algorithm 4.2 p109
 */
cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(2*N,9,CV_32F);

/*
 *  1点 变成 2点   p2   =  H21 * p1
      u2         h1  h2  h3       u1
      v2  =      h4  h5  h6    *  v1
      1          h7  h8  h9       1

     或是使用叉乘 得到0    * x = H y ，则对向量 x和Hy 进行叉乘为0，即：
					* | 0 -1  v2|    |h1 h2 h3|      |u1|     |0|
					* | 1  0 -u2| *  |h4 h5 h6| *    |v1| =  |0|
					* |-v2  u2 0|    |h7 h8 h9|      |1 |     |0|


      u2 = (h1*u1 + h2*v1 + h3) /( h7*u1 + h8*v1 + h9)
      v2 = (h4*u1 + h5*v1 + h6) /( h7*u1 + h8*v1 + h9)

      -((h4*u1 + h5*v1 + h6) - ( h7*u1*v2 + h8*v1*v2 + h9*v2))=0  式子为0  左侧加 - 号不变
        h1*u1 + h2*v1 + h3 - ( h7*u1*u2 + h8*v1*u2 + h9*u2)=0

        0    0   0  -u1  -v1  -1   u1*v2   v1*v2    v2
        u1 v1  1    0    0    0   -u1*u2  - v1*u2  -u2    ×(h1 h2 h3 h4 h5 h6 h7 h8 h9)转置  = 0

        8对点  约束 A
        A × h = 0 求h   奇异值分解 A 得到 单元矩阵 H
 */
    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -u1;
        A.at<float>(2*i,4) = -v1;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2;

        A.at<float>(2*i+1,0) = u1;
        A.at<float>(2*i+1,1) = v1;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2;

    }

    cv::Mat u,w,vt;
    // A × h = 0 求h
    // 在matlab中，[U,S,V]=svd(A)，其中U和V代表二个相互正交矩阵，而S代表一对角矩阵。
    //和QR分解法相同者， 原矩阵A不必为正方矩阵。
    //使用SVD分解法的用途是解最小平方误差法和数据压缩。
    // cv::SVDecomp(A,S,U,VT,SVD::FULL_UV);  //后面的FULL_UV表示把U和VT补充称单位正交方阵;
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    return vt.row(8).reshape(0, 3);// v的最后一列
}

// 通过svd进行最小二乘求解

// 8 对点 每个点 提供 一个约束
//8个点对 得到八个约束
//A *f = 0 求 f   奇异值分解 得到 f
/**
	*      构建基础矩阵的约束方程，给定一对点对应m=(u1,v1,1)T, m'=(u2,v2,1)T
	*  	   满足基础矩阵F   m'T F m=0,令F=(f_ij),则约束方程可以化简为：
	*  	    u2u1 f_11 + u2v1 f_12 + u2 f_13+v2u1f_21+v2v1f_22+v2f_23+u1f_31+v1f_32+f_33=0
	*  	    令f = (f_11,f_12,f_13,f_21,f_22,f_23,f_31,f_32,f_33)
	*  	    则(u2u1,u2v1,u2,v2u1,v2v1,v2,u1,v1,1)f=0;
	*  	    这样，给定N个对应点就可以得到线性方程组Af=0
	*  	    A就是一个N*9的矩阵，由于基础矩阵是非零的，所以f是一个非零向量，即
	*  	    线性方程组有非零解，另外基础矩阵的秩为2，重要的约束条件
	*/

// x'Fx = 0 整理可得：Af = 0
// A = | x'x x'y x' y'x y'y y' x y 1 |, f = | f1 f2 f3 f4 f5 f6 f7 f8 f9 |
// 通过SVD求解Af = 0，A'A最小特征值对应的特征向量即为解
/**
 * @brief 从特征点匹配求fundamental matrix（normalized 8点法）
 * @param  vP1 归一化后的点, in reference frame
 * @param  vP2 归一化后的点, in current frame
 * @return     基础矩阵
 * @see          Multiple View Geometry in Computer Vision - Algorithm 11.1 p282 (中文版 p191)
 */
cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(N,9,CV_32F);
/*
 *  * p2------> p1
 *                    f1   f2    f3     u1
 *   (u2 v2 1)    *   f4   f5    f6  *  v1    = 0  应该=0 不等于零的就是误差
 * 		              f7   f8    f9	    1
 * 	a1 = f1*u2 + f4*v2 + f7;
	b1 = f2*u2 + f5*v2 + f8;
	c1 =  f3*u2 + f6*v2 + f9;

       a1*u1+ b1*v1+ c1= 0
      一个点对 得到一个约束方程
       f1*u1*u2 + f2*v1*u2  + f3*u2 + f4*u1*v2  + f5*v1*v2 + f6*v2 +  f7*u1 + f8*v1 + f9 =0

     [  u1*u2   v1*u2   u2   u1*v2    v1*v2    v2  u1  v1 1 ] * [f1 f2 f3 f4 f5 f6 f7 f8 f9]转置  = 0

     8个点对 得到八个约束

     A *f = 0 求 f   奇异值分解得到F 基础矩阵 且其秩为2 需要再奇异值分解 后 取对角矩阵 秩为2 在合成F

 */
    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat Fpre = vt.row(8).reshape(0, 3);// F 基础矩阵的秩为2，需要在分解后，取对角矩阵，秩为2，再合成F

    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<float>(2)=0;//  基础矩阵的秩为2，重要的约束条件

    return  u*cv::Mat::diag(w)*vt;// 再合成F
}

// 计算单应矩阵 得分
/*
 * 【3】变换矩阵 评分 方法
 *  SM=∑i( ρM( d2cr(xic,xir,M)   +   ρM( d2rc(xic,xir,M ) )
 *  d2cr 为2D-2D点对，通过转换矩阵的对称转换误差
 *
 *  ρM 函数为 ρM(d^2)  = 0         当  d^2 > 阈值(单应矩阵时 为 5.991  基础矩阵时为 3.84)
 *                   阈值 - d^2    当  d^2 < 阈值
 *
 */

/**
 * @brief 对给定的homography matrix打分
 *
 * @see
 * - Author's paper - IV. AUTOMATIC MAP INITIALIZATION （2）
 * - Multiple View Geometry in Computer Vision - symmetric transfer errors: 4.2.2 Geometric distance
 * - Multiple View Geometry in Computer Vision - model selection 4.7.1 RANSAC
 */
float Initializer::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
{   
    const int N = mvMatches12.size();
    // |h11 h12 h13|
    // |h21 h22 h23|
    // |h31 h32 h33|
    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);
    // |h11inv h12inv h13inv|
    // |h21inv h22inv h23inv|
    // |h31inv h32inv h33inv|
    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
    const float th = 5.991;

    //信息矩阵，方差平方的倒数
    const float invSigmaSquare = 1.0/(sigma*sigma);//sigma=1.0

    // N对特征匹配点
    for(int i=0; i<N; i++)//计算单应矩阵 变换 每个点对时产生的对称转换误差
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];
/*
 *
* 1点 变成 2点
u2         h11  h12  h13       u1
v2  =      h21  h22  h23    *  v1
1          h31  h32  h33        1   第三行

* 2 点 变成 1点
u1‘        h11inv   h12inv   h13inv        u2
v1’  =     h21inv   h22inv   h23inv     *  v2
1          h31inv   h32inv   h33inv        1    第三行 h31inv*u2+h32inv*v2+h33inv
前两行同除以第三行，消去非零因子
p2 由单应转换到 p1
u1‘ = (h11inv*u2+h12inv*v2+h13inv)* 第三行倒数
v1’ = (h21inv*u2+h22inv*v2+h23inv)*第三行倒数
然后计算和真实p1点坐标的差值
(u1-u2in1)*(u1-u2in1) + (v1-v2in1)*(v1-v2in1)   横纵坐标差值平方和
 */
// 步骤1： p2 由单应转换到p1的距离误差以及得分
        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in first image
        // x2in1 = H12*x2
        // 将图像2中的特征点单应到图像1中
        // |u1|   |h11inv h12inv h13inv||u2|
        // |v1| = |h21inv h22inv h23inv||v2|
        // |1 |   |h31inv h32inv h33inv||1 |
        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;// p2 由单应转换到 p1‘
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;
        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);// 横纵坐标差值平方和
        const float chiSquare1 = squareDist1*invSigmaSquare;// 根据方差归一化误差

        if(chiSquare1>th)//距离大于阈值，说明该点变换的效果差
            bIn = false;
        else
            score += th - chiSquare1;// 阈值 - 距离差值 ，得到得分，差值越小，得分越高

// 步骤2：p1由单应转换到p2 距离误差以及得分
        // Reprojection error in second image
        // x1in2 = H21*x1   p1点 变成p2点 误差
        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;
        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);// 计算重投影误差
        const float chiSquare2 = squareDist2*invSigmaSquare;// 根据方差归一化误差

        if(chiSquare2>th)
            bIn = false;
        else
            score += th - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;// 是内点  误差较小
        else
            vbMatchesInliers[i]=false;// 是野点 误差较大
    }

    return score;
}

/**
 * @brief 对给定的fundamental matrix打分
 * p2 转置 * F21 * p1 = 0
 * F21 * p1为 帧1关键点p1在帧2上的极线l1
 *
 *
 *  p2 应该在这条极限附近 求p2到极线l的距离，可以作为误差
 * 	 极线l：ax + by + c = 0
 * 	 (u,v)到l的距离为：d = |au+bv+c| / sqrt(a^2+b^2)
 * 	 d^2 = |au+bv+c|^2/(a^2+b^2)
 *
 * p2 转置 * F21 为帧2关键点p2在帧1上的极线 l2
 *
 *
 * @see
 * - Author's paper - IV. AUTOMATIC MAP INITIALIZATION （2）
 * - Multiple View Geometry in Computer Vision - symmetric transfer errors: 4.2.2 Geometric distance
 * - Multiple View Geometry in Computer Vision - model selection 4.7.1 RANSAC
 */
float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
{
    const int N = mvMatches12.size();

    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
    const float th = 3.841;
    const float thScore = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);//信息矩阵，方差平方的倒数

    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;
//  p1 ------> p2 误差 得分------------------------------
        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)
        // F21x1可以算出x1在图像2中x2对应的线l
        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

        // x2应该在l这条线上:x2点乘l = 0
        // 计算x2特征点到 极线 的距离：
        // 极线l：ax + by + c = 0
        // (u,v)到l的距离为：d = |au+bv+c| / sqrt(a^2+b^2)
        // d^2 = |au+bv+c|^2/(a^2+b^2)
        const float num2 = a2*u2+b2*v2+c2;
        const float squareDist1 = num2*num2/(a2*a2+b2*b2);// 点到线的几何距离 的平方
        const float chiSquare1 = squareDist1*invSigmaSquare;// 根据方差归一化误差

        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)
        // l1 =x2转置 × F21=(a1,b1,c1)
//  p2 ------> p1 误差 得分-------------------------
        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;
        const float num1 = a1*u1+b1*v1+c1;
        const float squareDist2 = num1*num1/(a1*a1+b1*b1);
        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;//内点  误差较小
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

/*
 从基本矩阵恢复 旋转矩阵R 和 平移向量t
 计算 本质矩阵 E  =  K转置逆 * F  * K
 从本质矩阵恢复 旋转矩阵R 和 平移向量t
 恢复四种假设 并验证
理论参考 Result 9.19 in Multiple View Geometry in Computer Vision
 */
//                          |0 -1  0|
// E = U Sigma V'   let W = |1  0  0| 为RZ(90)  绕Z轴旋转 90度（x变成原来的y y变成原来的-x z轴没变）
//                          |0  0  1|
// 得到4个解 E = [R|t]
// R1 = UWV' R2 = UW'V' t1 = U3 t2 = -U3

/**
 * @brief 从基本矩阵 F 恢复R t
 *
 * 度量重构
 * 1. 由Fundamental矩阵结合相机内参K，得到Essential矩阵: \f$ E = k转置F k \f$
 * 2. SVD分解得到R t
 * 3. 进行cheirality check, 从四个解中找出最合适的解
 *
 * @see Multiple View Geometry in Computer Vision - Result 9.19 p259
 */
bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                            cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;// 符合基本矩阵F的内点数量

// 步骤1： 计算 本质矩阵 E  =  K转置 * F  * K
    // Compute Essential Matrix from Fundamental Matrix
    cv::Mat E21 = K.t()*F21*K;
    cv::Mat R1, R2, t;

// 步骤2：  从本质矩阵恢复 旋转矩阵R 和 平移向量t
        /*
        *  对 本质矩阵E 进行奇异值分解，得到可能的解
        * t = u * RZ(90) * u转置
        * R= u * RZ(90) * V转置
        * 组合情况有四种
        */
    // Recover the 4 motion hypotheses
    // 虽然这个函数对t有归一化，但并没有决定单目整个SLAM过程的尺度
    // 因为CreateInitialMapMonocular函数对3D点深度会缩放，然后反过来对 t 有改变
    DecomposeE(E21,R1,R2,t);  

    cv::Mat t1=t;
    cv::Mat t2=-t;

// 步骤3： 恢复四种假设 并验证 Reconstruct with the 4 hyphoteses and check
        // 这4个解中只有一个是合理的，可以使用可视化约束来选择，
        // 与单应性矩阵做sfm一样的方法，即将4种解都进行三角化，然后从中选择出最合适的解。
    // Reconstruct with the 4 hyphoteses and check
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
    float parallax1,parallax2, parallax3, parallax4;

    int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);

    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

    R21 = cv::Mat();
    t21 = cv::Mat();

    // minTriangulated为可以三角化恢复三维点的个数
    int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);

    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    if(maxGood<nMinGood || nsimilar>1)
    {// 四个结果中如果没有明显的最优结果，则返回失败
        return false;
    }

    // 取比较大的视差角  四种假设
    // If best reconstruction has enough parallax initialize
    if(maxGood==nGood1)
    {
        if(parallax1>minParallax)
        {
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood2)
    {
        if(parallax2>minParallax)
        {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood3)
    {
        if(parallax3>minParallax)
        {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood4)
    {
        if(parallax4>minParallax)
        {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    return false;
}

/*
 从单应矩阵恢复 旋转矩阵R 和 平移向量t
 理论参考
  // Faugeras et al, Motion and structure from motion in a piecewise planar environment.
  International Journal of Pattern Recognition and Artificial Intelligence, 1988.

 https://hal.archives-ouvertes.fr/inria-00075698/document

p2   =  H21 * p1
p2 = K( RP + t)  = KTP = H21 * KP
T =  K 逆 * H21*K
在求得单应性变化H后，本文使用FAUGERAS的论文[1]的方法，提取8种运动假设。
这个方法通过可视化约束来测试选择合理的解。但是如果在低视差的情况下，
点云会跑到相机的前面或后面，测试就会出现错误从而选择一个错误的解。
文中使用的是直接三角化 8种方案，检查两个相机前面具有较少的重投影误差情况下，
在视图低视差情况下是否大部分云点都可以看到。如果没有一个解很合适，就不执行初始化，
重新从第一步开始。这种方法在低视差和两个交叉的视图情况下，初始化程序更具鲁棒性。
 */
// H矩阵分解常见有两种方法：Faugeras SVD-based decomposition 和 Zhang SVD-based decomposition
// 参考文献：Motion and structure from motion in a piecewise plannar environment
// 这篇参考文献和下面的代码使用了Faugeras SVD-based decomposition算法

/**
 * @brief 从H恢复R t
 *
 * @see
 * - Faugeras et al, Motion and structure from motion in a piecewise planar environment. International Journal of Pattern Recognition and Artificial Intelligence, 1988.
 * - Deeper understanding of the homography decomposition for vision-based control
 */
bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;//匹配点对内点个数

    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.
    // International Journal of Pattern Recognition and Artificial Intelligence, 1988

    // 因为特征点是图像坐标系，所以将H矩阵由相机坐标系换算到图像坐标系
    cv::Mat invK = K.inv();
    cv::Mat A = invK*H21*K;

    cv::Mat U,w,Vt,V;
    cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
    V=Vt.t();

    float s = cv::determinant(U)*cv::determinant(Vt);

    float d1 = w.at<float>(0);
    float d2 = w.at<float>(1);
    float d3 = w.at<float>(2);

    // SVD分解的正常情况是特征值降序排列
    if(d1/d2<1.00001 || d2/d3<1.00001)
    {
        return false;
    }

    vector<cv::Mat> vR, vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    // 法向量n'= [x1 0 x3] 对应ppt的公式17
    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
    float x1[] = {aux1,aux1,-aux1,-aux1};
    float x3[] = {aux3,-aux3,aux3,-aux3};

    //case d'=d2
    // 计算ppt中公式19
    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};
    // 计算旋转矩阵 R‘，计算ppt中公式18
    //          | ctheta         0   -aux_stheta|         | aux1|
    // Rp =     |    0               1       0  |  tp =   |  0     |
    //          | aux_stheta  0    ctheta       |         |-aux3|

    //          | ctheta          0    aux_stheta|          | aux1|
    // Rp =     |    0            1       0      |  tp =    |  0  |
    //          |-aux_stheta  0    ctheta        |          | aux3|

    //          | ctheta         0    aux_stheta|         |-aux1|
    // Rp =     |    0             1       0    |  tp =  |  0     |
    //          |-aux_stheta  0    ctheta       |         |-aux3|

    //          | ctheta         0   -aux_stheta|         |-aux1|
    // Rp = |    0               1       0      |  tp =   |  0  |
    //          | aux_stheta  0    ctheta       |          | aux3|
    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=ctheta;
        Rp.at<float>(0,2)=-stheta[i];
        Rp.at<float>(2,0)=stheta[i];
        Rp.at<float>(2,2)=ctheta;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=-x3[i];
        tp*=d1-d3;

        // 这里虽然对t有归一化，并没有决定单目整个SLAM过程的尺度
        // 因为CreateInitialMapMonocular函数对3D点深度会缩放，然后反过来对 t 有改变
        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    //case d'=-d2
    // 计算ppt中公式22
    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};
    // 计算旋转矩阵 R‘，计算ppt中公式21
    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=cphi;
        Rp.at<float>(0,2)=sphi[i];
        Rp.at<float>(1,1)=-1;
        Rp.at<float>(2,0)=sphi[i];
        Rp.at<float>(2,2)=-cphi;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=x3[i];
        tp*=d1+d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }


    int bestGood = 0;
    int secondBestGood = 0;    
    int bestSolutionIdx = -1;
    float bestParallax = -1;
    vector<cv::Point3f> bestP3D;
    vector<bool> bestTriangulated;

    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
    // d'=d2和d'=-d2分别对应8组(R t)
    for(size_t i=0; i<8; i++)
    {
        float parallaxi;
        vector<cv::Point3f> vP3Di;
        vector<bool> vbTriangulatedi;
        int nGood = CheckRT(vR[i],vt[i],mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K,vP3Di, 4.0*mSigma2, vbTriangulatedi, parallaxi);
        // 保留最优的和次优的
        if(nGood>bestGood)
        {
            secondBestGood = bestGood;
            bestGood = nGood;
            bestSolutionIdx = i;
            bestParallax = parallaxi;
            bestP3D = vP3Di;
            bestTriangulated = vbTriangulatedi;
        }
        else if(nGood>secondBestGood)
        {
            secondBestGood = nGood;
        }
    }


    if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
    {
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        vP3D = bestP3D;
        vbTriangulated = bestTriangulated;

        return true;
    }

    return false;
}

/*
 * 三角化得到3D点
 *  *三角测量法 求解 两组单目相机  图像点深度
 * s1 * x1 = s2  * R * x2 + t
 * x1 x2 为两帧图像上 两点对 在归一化坐标平面上的坐标 k逆* p
 * s1  和 s2为两个特征点的深度 ，由于误差存在， s1 * x1 = s2  * R * x2 + t不精确相等
 * 常见的是求解最小二乘解，而不是零解
 *  s1 * x1叉乘x1 = s2 * x1叉乘* R * x2 + x1叉乘 t=0 可以求得x2
 *
 */
/*
 平面二维点摄影矩阵到三维点  P1 = K × [I 0]    P2 = K * [R  t]
  kp1 = P1 * p3dC1       p3dC1  特征点匹配对 对应的 世界3维点
  kp2 = P2 * p3dC1
  kp1 叉乘  P1 * p3dC1 =0
  kp2 叉乘  P2 * p3dC1 =0
 p = ( x,y,1)
 其叉乘矩阵为
     //  叉乘矩阵 = [0  -1  y;
    //              1   0  -x;
    //              -y   x  0 ]
  一个方程得到两个约束
  对于第一行 0  -1  y; 会与P的三行分别相乘 得到四个值 与齐次3d点坐标相乘得到 0
  有 (y * P.row(2) - P.row(1) ) * D =0
      (-x *P.row(2) + P.row(0) ) * D =0 ===> (x *P.row(2) - P.row(0) ) * D =0
    两个方程得到 4个约束
    A × D = 0
    对A进行奇异值分解 求解线性方程 得到 D  （D是3维齐次坐标，需要除以第四个尺度因子 归一化）
 */
// Trianularization: 已知匹配特征点对{x x'} 和 各自相机矩阵{P P'}, 估计三维点 X
// x' = P'X  x = PX
// 它们都属于 x = aPX模型
//                            |X|
// |x|     |p1 p2   p3  p4|   |Y|     |x|    |--p0--||.|
// |y| = a |p5 p6   p7  p8|   |Z| ===>|y| = a|--p1--||X|
// |z|     |p9 p10 p11 p12|   |1|     |z|    |--p2--||.|
// 采用DLT的方法：x叉乘PX = 0
// |yp2 -  p1|      |0|
// |p0  -  xp2| X = |0|
// |xp1 - yp0|      |0|
// 两个点:
// |yp2   -  p1  |     |0|
// |p0    -  xp2 | X = |0| ===> AX = 0
// |y'p2' -  p1' |     |0|
// |p0'   - x'p2'|     |0|
// 变成程序中的形式：
// |xp2  - p0 |     |0|
// |yp2  - p1 | X = |0| ===> AX = 0
// |x'p2'- p0'|     |0|
// |y'p2'- p1'|     |0|
/**
 * @brief 给定投影矩阵P1,P2和图像上的点kp1,kp2，从而恢复3D坐标
 *
 * @param kp1 特征点, in reference frame
 * @param kp2 特征点, in current frame
 * @param P1  投影矩阵P1
 * @param P2  投影矩阵P２
 * @param x3D 三维点
 * @see       Multiple View Geometry in Computer Vision - 12.2 Linear triangulation methods p312
 */
void Initializer::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    cv::Mat A(4,4,CV_32F);
    // 在DecomposeE函数和ReconstructH函数中对t有归一化
    // 这里三角化过程中恢复的3D点深度取决于 t 的尺度，
    // 但是这里恢复的3D点并没有决定单目整个SLAM过程的尺度
    // 因为CreateInitialMapMonocular函数对3D点深度会缩放，然后反过来对 t 有改变
    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}

void Initializer::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();

    vNormalizedPoints.resize(N);

    for(int i=0; i<N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    meanX = meanX/N;
    meanY = meanY/N;

    float meanDevX = 0;
    float meanDevY = 0;

    // 将所有vKeys点减去中心坐标，使x坐标和y坐标均值分别为0
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    // 将x坐标和y坐标分别进行尺度缩放，使得x坐标和y坐标的一阶绝对矩分别为1
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    /*
    计算并返回标准化矩阵
    ui' = (ui - mean_x) * sX =  ui * sX + vi * 0  + (-mean_x * sX) * 1
    vi' = (vi - mean_y) * sY =  ui * 0  + vi * sY + (-mean_y * sY) * 1
    1   =                       ui * 0  + vi * 0  +      1         * 1

    可以得到：
    ui'     sX  0   (-mean_x * sX)      ui
    vi' =   0   sY  (-mean_y * sY)   *  vi
    1       0   0        1              1
    标准化后的的坐标 = 标准化矩阵T * 原坐标
    所以标准化矩阵:
    T =  sX  0   (-mean_x * sX)
    0   sY  (-mean_y * sY)
    0   0        1
    而由标准化坐标 还原 回 原坐标(左乘T 逆)：
    原坐标 = 标准化矩阵T 逆矩阵 * 标准化后的的坐标

    再者用于还原 单应矩阵 下面需要用到：
    p1'  ------> Hn -------> p2'   , p2'   = Hn*p1'
    T1*p1 -----> Hn -------> T2*p2 , T2*p2 = Hn*(T1*p1)
    左乘 T2逆 ，得到   p2 = T2逆 * Hn*(T1*p1)= H21i*p1
    H21i = T2逆 * Hn * T1
    */
    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}

/*
 * 检查求得的R t 是否符合
 * 接受 R,t ，一组成功的匹配。最后给出的结果是这组匹配中有多少匹配是
 * 能够在这组 R,t 下正确三角化的（即 Z都大于0），并且输出这些三角化之后的三维点。
如果三角化生成的三维点 Z小于等于0，且三角化的“前方交会角”（余弦是 cosParallax）不会太小，
那么这个三维点三角化错误，舍弃。
通过了 Z的检验，之后将这个三维点分别投影到两张影像上，
计算投影的像素误差，误差大于2倍中误差，舍弃。
 */
/**
 * @brief 进行cheirality check，从而进一步找出F分解后最合适的解
 */
int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{
    // Calibration parameters
    // 校正参数,内参
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    vbGood = vector<bool>(vKeys1.size(),false);
    vP3D.resize(vKeys1.size());

    vector<float> vCosParallax;
    vCosParallax.reserve(vKeys1.size());

    // Camera 1 Projection Matrix K[I|0]
// 步骤1：得到一个相机的投影矩阵
    // 以第一个相机的光心作为世界坐标系
    // 相机1  变换矩阵 在第一幅图像下 的变换矩阵  Pc1  =   Pw  =  T1 * Pw      T1 = [I|0]
    // Pp1  = K *  Pc1 = K * T1 * Pw  =   [K|0] *Pw  = P1 × Pw
    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
    K.copyTo(P1.rowRange(0,3).colRange(0,3));
    // 第一个相机的光心在世界坐标系下的坐标
    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);// 相机1原点 000

// 步骤2：得到第二个相机的投影矩阵
    // Camera 2 Projection Matrix K[R|t]
    // 相机2  变换矩阵  Pc2  =   Pw  =  T2 * Pw      T2 = [R|t]
    // Pp2  = K *  Pc2 = K * T2 * Pw  =  K* [R|t] *Pw  = P2 × Pw
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = K*P2;

    // 第二个相机的光心在世界坐标系下的坐标
    cv::Mat O2 = -R.t()*t;//相机2原点  R逆 * - t  R 为正交矩阵  逆 = 转置

    int nGood=0;


    for(size_t i=0, iend=vMatches12.size();i<iend;i++)// 每一个匹配点对
    {
        if(!vbMatchesInliers[i])// 离线点  非内点
            continue;

        // kp1和kp2是匹配特征点
        const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
        const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
        cv::Mat p3dC1;


// 步骤3：利用三角法恢复三维点p3dC1
        // kp1 = P1 * p3dC1     kp2 = P2 * p3dC1
        Triangulate(kp1,kp2,P1,P2,p3dC1);

        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {// 求出的3d点坐标 值有效
            vbGood[vMatches12[i].first]=false;
            continue;
        }

// 步骤4：计算视差角余弦值
        // Check parallax
        cv::Mat normal1 = p3dC1 - O1;
        float dist1 = cv::norm(normal1);
        cv::Mat normal2 = p3dC1 - O2;
        float dist2 = cv::norm(normal2);
        float cosParallax = normal1.dot(normal2)/(dist1*dist2);

// 步骤5：判断3D点是否在两个摄像头前方
        // 步骤5.1：3D点深度为负，在第一个摄像头后方，淘汰
        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // 步骤5.2：3D点深度为负，在第二个摄像头后方，淘汰
        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R*p3dC1+t;
        if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

// 步骤6：计算重投影误差
        // 计算3D点在第一个图像上的投影误差
        // Check reprojection error in first image
        float im1x, im1y;
        float invZ1 = 1.0/p3dC1.at<float>(2);
        im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<float>(1)*invZ1+cy;
        float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);

        // 步骤6.1：重投影误差太大，跳过淘汰
        // 一般视差角比较小时重投影误差比较大
        if(squareError1>th2)
            continue;

        // 计算3D点在第二个图像上的投影误差
        // Check reprojection error in second image
        float im2x, im2y;
        float invZ2 = 1.0/p3dC2.at<float>(2);
        im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
        im2y = fy*p3dC2.at<float>(1)*invZ2+cy;
        float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);

        // 步骤6.2：重投影误差太大，跳过淘汰
        // 一般视差角比较小时重投影误差比较大
        if(squareError2>th2)
            continue;

// 步骤7：统计经过检验的3D点个数，记录3D点视差角
        vCosParallax.push_back(cosParallax);
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
        //TODO qzc
        nGood++;

        if(cosParallax<0.99998){
            vbGood[vMatches12[i].first]=true;
            //TODO qzc
            //nGood++;
        }
    }

// 步骤8：得到3D点中较大的视差角
    if(nGood>0)
    {
        sort(vCosParallax.begin(),vCosParallax.end());

        // trick! 排序后并没有取最大的视差角
        // 取一个较大的视差角
        size_t idx = min(50,int(vCosParallax.size()-1));
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;

    return nGood;
}

/*
 * 从本质矩阵恢复 旋转矩阵R 和 平移向量t
 *  对 本质矩阵E 进行奇异值分解   得到可能的解
 * t = u * RZ(90) * u转置
 * R= u * RZ(90) * V转置
 * 组合情况有四种
 */

/**
 * @brief 分解Essential矩阵
 *
 * F矩阵通过结合内参可以得到Essential矩阵，分解E矩阵将得到4组解 \n
 * 这4组解分别为[R1,t],[R1,-t],[R2,t],[R2,-t]
 * @param E  Essential Matrix
 * @param R1 Rotation Matrix 1
 * @param R2 Rotation Matrix 2
 * @param t  Translation
 * @see Multiple View Geometry in Computer Vision - Result 9.19 p259
 */
void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    // 【1】对 本质矩阵E 进行奇异值分解
    cv::Mat u,w,vt;
    cv::SVD::compute(E,w,u,vt);// 其中u和v代表二个相互正交矩阵，而w代表一对角矩阵

    // 对 t 有归一化，但是这个地方并没有决定单目整个SLAM过程的尺度
    // 因为CreateInitialMapMonocular函数对3D点深度会缩放，然后反过来对 t 有改变
    u.col(2).copyTo(t);
    t=t/cv::norm(t);

    // 沿着Z轴旋转 90度得到的旋转矩阵（逆时针为正方向）
    // z 轴还是 原来的z轴，y轴变成原来的x轴的负方向，x轴变成原来的y轴
    // 所以旋转矩阵为
    //          0  -1   0
    //		    1   0   0
    //		    0   0   1
    // 沿着Z轴旋转-90度
    // z 轴还是原来的z轴，y轴变成原来的x轴，x轴变成原来的y轴的负方向
    // 所以 旋转矩阵  为 0   1   0  为上 旋转矩阵的转置矩阵
    //		          -1   0   0
    //		           0   0   1

    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    R1 = u*W*vt;
    if(cv::determinant(R1)<0)
        R1=-R1;

    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;
}

} //namespace ORB_SLAM
