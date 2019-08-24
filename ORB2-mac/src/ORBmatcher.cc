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

#include "ORBmatcher.h"

#include<limits.h>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint.h>

using namespace std;

namespace ORB_SLAM2
{

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}

// 当前帧和局部地图之间的匹配
// 最好的匹配和次好的匹配在同一金字塔层级,并且最短的距离不小于次短距离的80%,不被选为匹配点
/**
* @brief 通过投影，对Local MapPoint进行跟踪
*
* 将Local MapPoint投影到当前帧中, 由此增加当前帧的MapPoints \n
* 在SearchLocalPoints()中已经将Local MapPoints重投影（isInFrustum()）到当前帧 \n
* 并标记了这些点是否在当前帧的视野中，即 mbTrackInView \n
* 对这些MapPoints，在其投影点附近根据描述子距离选取匹配，
* 以及最终的方向投票机制进行剔除
* @param  F           当前帧
* @param  vpMapPoints Local MapPoints 局部地图点和当前帧有关连的帧对应的地图点集合
* @param  th          搜索窗口大小尺寸尺度
* @return             成功匹配的数量
* @see SearchLocalPoints() isInFrustum()
*/
int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th)
{
    int nmatches=0;

    const bool bFactor = th!=1.0;

    for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)//局部地图点
    {
        MapPoint* pMP = vpMapPoints[iMP];
// 步骤1 ： 判断该点是否要投影
        if(!pMP->mbTrackInView)//不在视野内
            continue;

        if(pMP->isBad())
            continue;
// 步骤2 ： 通过距离预测的金字塔层数，该层数相对于当前的帧
        const int &nPredictedLevel = pMP->mnTrackScaleLevel;

// 步骤3 ： 搜索窗口的大小取决于视角, 若当前视角和平均视角夹角接近0度时, r取一个较小的值
        // The size of the window will depend on the viewing direction
        float r = RadiusByViewingCos(pMP->mTrackViewCos);
        // 如果需要进行更粗糙的搜索，则增大范围
        if(bFactor)
            r*=th;

// 步骤4： 通过投影点(投影到当前帧,见isInFrustum())以及搜索窗口和预测的尺度进行搜索, 找出附近的兴趣点
        const vector<size_t> vIndices =
                F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

        if(vIndices.empty())
            continue;

        const cv::Mat MPdescriptor = pMP->GetDescriptor();// 局部地图点的描述子

        int bestDist=256;
        int bestLevel= -1;
        int bestDist2=256;
        int bestLevel2 = -1;
        int bestIdx =-1 ;

// 步骤5： 地图点描述子和当前帧候选关键点描述子匹配
        // Get best and second matches with near keypoints
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
// 步骤6：  当前帧关键点已经有对应的地图点 或者 地图点计算出来的匹配点y偏移比当前的立体匹配点误差过大跳过
            // 如果当前帧Frame中的该兴趣点已经有对应的MapPoint了,则退出该次循环
            if(F.mvpMapPoints[idx])
                if(F.mvpMapPoints[idx]->Observations()>0)
                    continue;

            // 跟踪到的匹配点坐标与实际立体匹配的误差过大跳过
            if(F.mvuRight[idx]>0)// 双目 / 深度相机
            {
                const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
                if(er>r*F.mvScaleFactors[nPredictedLevel])// 跟踪到的 匹配点坐标 与实际立体匹配的 误差过大
                    continue;
            }

            const cv::Mat &d = F.mDescriptors.row(idx);// 每一个候选匹配点的描述子
            const int dist = DescriptorDistance(MPdescriptor,d);// 局部地图点 与 当前帧地图点 之间的 描述子距离
// 步骤7：根据描述子距离 寻找 距离最小和次小的特征点
            if(dist<bestDist)
            {
                bestDist2=bestDist;// 次近的距离
                bestDist=dist;// 最近的距离
                bestLevel2 = bestLevel;
                bestLevel = F.mvKeysUn[idx].octave;// 对应关键点的金字塔层级
                bestIdx=idx;
            }
            else if(dist<bestDist2)
            {
                bestLevel2 = F.mvKeysUn[idx].octave;
                bestDist2=dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        if(bestDist<=TH_HIGH)
        {
            // 最好的匹配和次好的匹配在同一金字塔层级,并且最短的距离不小于次短距离的80%
            if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                continue;
// 步骤7：为Frame中的兴趣点增加对应的MapPoint
            F.mvpMapPoints[bestIdx]=pMP;
            nmatches++;
        }
    }

    return nmatches;
}

float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998)
        return 2.5;
    else
        return 4.0;
}


bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const KeyFrame* pKF2)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;

    const float dsqr = num*num/den;

    return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
}

// 当前帧和参考关键帧中的地图点进行特征匹配，匹配到已有地图点
// 关键帧和当前帧均用字典单词线性表示
// 对应单词的描述子，肯定比较相近，取对应单词的描述子进行匹配可以加速匹配
// 当前帧每个关键点的描述子 和 参考关键帧每个地图点的描述子匹配
// 保留距离最近的匹配地图点 且最短距离和次短距离相差不大 （ mfNNratio）
// 如果需要考虑关键点的方向信息
// 统计当前帧 关键点的方向 到30步长 的方向直方图
// 保留方向直方图中最高的三个bin中 关键点 匹配的 地图点  匹配点对
/**
* @brief 通过词包，对参考关键帧的地图点进行跟踪
*
* 通过bow对pKF和F中的点描述子 进行快速匹配（不属于同一node(词典单词)的特征点直接跳过匹配） \n
* 对属于同一node(词典单词)的特征点通过描述子距离进行匹配 \n
* 根据匹配，用参考关键帧pKF中特征点对应的MapPoint更新 当前帧F 中特征点对应的MapPoints \n
* 每个特征点都对应一个MapPoint，因此pKF中每个特征点的MapPoint也就是F中对应点的MapPoint \n
* 通过 距离阈值、比例阈值 和 角度投票进行剔除误匹配
* @param  pKF            KeyFrame           参考关键帧
* @param  F                 Current Frame  当前帧
* @param  vpMapPointMatches  当前帧 F中关键点 匹配到的地图点MapPoints ，NULL表示未匹配
* @return                   成功匹配的数量
*/
int ORBmatcher::SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)
{
    // 参考关键帧 的地图点
    const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();
    // 当前帧关键点个数匹配点 (对应原关键帧中的地图点)
    vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));
    // 参考关键帧的地图点描述子的特征向量
    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

    int nmatches=0;

    vector<int> rotHist[HISTO_LENGTH];// 方向向量直方图 HISTO_LENGTH=30
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    // 关键帧和当前帧均用字典单词线性表示
    // 对应单词的描述子肯定比较相近，取对应单词的描述子进行匹配可以加速匹配
    // 将属于同一节点(特定层)的ORB特征进行匹配
    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();// 参考关键帧特征点描述子词典特征向量，开始标志
    DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();// 当前帧特征点描述子词典特征向量，开始标志
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
    DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

    while(KFit != KFend && Fit != Fend)
    {
//步骤1：分别取出属于同一node的ORB特征点(只有属于同一node(单词)，才有可能是匹配点)
        if(KFit->first == Fit->first)// 同一个单词下的 描述子
        {
            const vector<unsigned int> vIndicesKF = KFit->second;
            const vector<unsigned int> vIndicesF = Fit->second;

// 步骤2：遍历关键帧KF中属于该node的地图点 其对应一个描述子
            for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)// 每一个参考 关键帧 地图点
            {
                const unsigned int realIdxKF = vIndicesKF[iKF];
                MapPoint* pMP = vpMapPointsKF[realIdxKF];// 取出KF中该特征对应的MapPoint
                if(!pMP)// 剔除 不好的地图点
                    continue;
                if(pMP->isBad())
                    continue;
                // 取出关键帧KF中该特征对应的描述子
                const cv::Mat &dKF= pKF->mDescriptors.row(realIdxKF);

                int bestDist1=256;// 最好的距离（最小距离）
                int bestIdxF =-1 ;
                int bestDist2=256;
// 步骤3：遍历当前帧 F 中属于该node的特征点，找到了最佳匹配点
                for(size_t iF=0; iF<vIndicesF.size(); iF++)//每一个当前帧
                {
                    const unsigned int realIdxF = vIndicesF[iF];
                    // 表明这个特征点点已经被匹配过了，不再匹配，加快速度
                    if(vpMapPointMatches[realIdxF])
                        continue;
                    // 取出当前帧 F 中该特征对应的描述子
                    const cv::Mat &dF = F.mDescriptors.row(realIdxF);
                    const int dist =  DescriptorDistance(dKF,dF);// 描述子之间的 距离
//  步骤4：找出最短距离和次短距离对应的 匹配点
                    if(dist<bestDist1)// dist < bestDist1 < bestDist2，更新bestDist1 bestDist2
                    {
                        bestDist2=bestDist1;// 次最短的距离
                        bestDist1=dist;// 最短的距离
                        bestIdxF=realIdxF;
                    }
                    else if(dist<bestDist2)// bestDist1 < dist < bestDist2，更新bestDist2
                    {
                        bestDist2=dist;
                    }
                }
// 步骤5：根据阈值 和 角度投票剔除误匹配
                if(bestDist1<=TH_LOW)// 最短的距离 小于阈值
                {
                    // trick!
                    // 最佳匹配比次佳匹配明显要好，那么最佳匹配才真正靠谱
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
// 步骤6：更新当前帧特征点对应的 地图点MapPoint
                        vpMapPointMatches[bestIdxF]=pMP;// 匹配到的 参考关键中 的地图点
                        const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];//地图点在 参考关键帧中的 像素点
                        if(mbCheckOrientation)// 查看方向是否 合适
                        {
                            // trick!
                            // angle：每个特征点在提取描述子时的旋转主方向角度，如果图像旋转了，这个角度将发生改变
                            // 所有的特征点的角度变化应该是一致的，通过直方图统计得到最准确的角度变化值
                            float rot = kp.angle-F.mvKeys[bestIdxF].angle;// 当前帧 的 关键点 的方向和匹配点方向 变化
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            // 对于每一对匹配点的角度差，均可以放入一个bin的范围内（360/HISTO_LENGTH）
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdxF);// 方向 直方图
                        }
                        nmatches++;
                    }
                }

            }

            KFit++;
            Fit++;
        }
        else if(KFit->first < Fit->first)
        {
            KFit = vFeatVecKF.lower_bound(Fit->first);
        }
        else
        {
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }

// 步骤7： 根据方向剔除误匹配的点
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;
        // 统计方向偏差直方图 频率最高的三个bin保留，其他范围内的匹配点剔除。
        // 另外，若最高的比第二高的高10倍以上，则只保留最高的bin中的匹配点。
        // 若最高的比第 三高的高10倍以上，则 保留最高的和第二高bin中的匹配点。
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            // 如果特征点的旋转角度变化量属于这三个组，则保留
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                // 将除了ind1 ind2 ind3以外的匹配点去掉
                vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}
/**
* @brief   为关键帧pKF中还没有匹配到3D地图点的2D特征点,从所给的地图点中匹配地图点
* 根据Sim3变换 转化到 欧式变换，
* 将每个vpPoints投影到 参考关键帧pKF的图像像素坐标系上，并根据尺度确定一个搜索区域， \n
        * 根据该MapPoint的描述子与该区域内的特征点进行匹配  \n
        * 如果匹配误差小于TH_LOW即匹配成功，更新vpMatched \n
* @param  pKF         KeyFrame            参考关键帧
* @param  Scw         参考关键帧的相似变换   [s*R t]
* @param  vpPoints    地图点
* @param  vpMatched   参考关键帧特征点对应的匹配点
* @param  th          匹配距离阈值
* @return             成功匹配的数量
*/
int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const vector<MapPoint*> &vpPoints, vector<MapPoint*> &vpMatched, int th)
{
    // 相机内参数, Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

// 步骤1：相似变换转换到欧式变换 归一化相似变换矩阵  Decompose Scw
    // | s*R  t|
    // |   0  1|
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);// 相似变换旋转矩阵
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));// 计算相似变换矩阵的尺度s
    cv::Mat Rcw = sRcw/scw;// 归一化的 旋转矩阵
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;//  归一化的 计算相似变换矩阵
    cv::Mat Ow = -Rcw.t()*tcw;// pKF坐标系下，世界坐标系到pKF的位移，方向由世界坐标系指向pKF
    // Rwc * twc  用来计算地图点距离相机的距离,进而推断在图像金字塔中可能的尺度

    // Set of MapPoints already found in the KeyFrame
// 步骤2： 使用set类型，并去除没有匹配的点，用于快速检索某个MapPoint是否有匹配
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

    int nmatches=0;

// 步骤3： 遍历所有的 地图点 MapPoints
    // For each Candidate MapPoint Project and Match
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

// 步骤4：地图点根据变换,转到当前帧相机坐标系下
        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();// 地图点的 世界坐标
        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;//  转到当前帧 相机坐标系下

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0)//剔除深度<0 的点
            continue;

// 步骤5：根据相机内参数 投影到 当前帧的图像像素坐标系下
        // Project into Image
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;
        const float u = fx*x+cx;
        const float v = fy*y+cy;
        // 地图点投影过来 如果不在图像范围内 就没有匹配点
        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

//步骤6：  判断距离是否在尺度协方差范围内 剔除
        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        //地图点距离相机的距离,进而推断在图像金字塔中可能的尺度 越远尺度小 越近尺度大
        cv::Mat PO = p3Dw-Ow;
        const float dist = cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();
        if(PO.dot(Pn)<0.5*dist)// 观察视角 必须小于 60度
            continue;

// 步骤7： 根据尺度确定搜索半径 进而在图像上确定候选关键点
        int nPredictedLevel = pMP->PredictScale(dist,pKF);//根据距离预测点处于的图像金字塔尺度

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);// 在图像上确定 候选 关键点

        if(vIndices.empty())
            continue;

//  步骤8：遍历候选关键点，地图点和关键帧上候选关键点进行描述子匹配，计算距离，保留最近距离的匹配
        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();
        int bestDist = 256;
        int bestIdx = -1;
        // 遍历搜索区域内所有特征点，与该MapPoint的描述子进行匹配
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(vpMatched[idx])//跳过  已经匹配上地图点 MapPoints 的像素点
                continue;

            const int &kpLevel= pKF->mvKeysUn[idx].octave;// 候选关键点 不在 由地图点预测的尺度到 最高尺度范围内 直接跳过
            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);// 计算距离 保存 最短的距离 对应的 关键点
            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_LOW)
        {
            vpMatched[bestIdx]=pMP;
            nmatches++;
        }

    }

    return nmatches;
}

int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
{
    int nmatches=0;
    vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);
    vector<int> vnMatches21(F2.mvKeysUn.size(),-1);

    for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
    {
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave;
        if(level1>0)//只取零层的特征点
            continue;

        vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);

        if(vIndices2.empty())
            continue;

        cv::Mat d1 = F1.mDescriptors.row(i1);

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
        {
            size_t i2 = *vit;

            cv::Mat d2 = F2.mDescriptors.row(i2);

            // 汉明距离
            int dist = DescriptorDistance(d1,d2);

            if(vMatchedDistance[i2]<=dist)
                continue;

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            }
            else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }

        // 详见SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)函数步骤4
        if(bestDist<=TH_LOW)
        {
            if(bestDist<(float)bestDist2*mfNNratio)
            {
                if(vnMatches21[bestIdx2]>=0)
                {
                    vnMatches12[vnMatches21[bestIdx2]]=-1;
                    nmatches--;
                }
                vnMatches12[i1]=bestIdx2;
                vnMatches21[bestIdx2]=i1;
                vMatchedDistance[bestIdx2]=bestDist;
                nmatches++;

                if(mbCheckOrientation)
                {
                    float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = round(rot*factor);
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                }
            }
        }

    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        // 取出直方图中值最大的三个index
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                int idx1 = rotHist[i][j];
                if(vnMatches12[idx1]>=0)
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }

    }

    //Update prev matched
    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
        if(vnMatches12[i1]>=0)
            vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;

    return nmatches;
}

    /**
    * @brief 通过词包，对关键帧的特征点进行跟踪，该函数用于闭环检测时两个关键帧间的特征点匹配
    *
    * 通过bow对pKF1和pKF2中的特征点进行快速匹配（不属于同一node(单词)的特征点直接跳过匹配） \n
    * 对属于同一node的特征点通过描述子距离进行匹配 \n
    * 根据匹配，更新vpMatches12 \n
    * 通过距离阈值、比例阈值和角度投票进行剔除误匹配
    * @param  pKF1               KeyFrame1
    * @param  pKF2               KeyFrame2
    * @param  vpMatches12        pKF2中与pKF1匹配的MapPoint，null表示没有匹配
    * @return                    成功匹配的数量
    */
int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
{
    const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;// 关键帧1 特征点
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec; // 关键帧1 特征点 词典描述向量
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();// 关键帧1 特征点 匹配的 地图点
    const cv::Mat &Descriptors1 = pKF1->mDescriptors;// 键帧1 特征点的 描述子 矩阵

    const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const cv::Mat &Descriptors2 = pKF2->mDescriptors;

    // 为关键帧1的地图点 初始化 匹配点
    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
    vector<bool> vbMatched2(vpMapPoints2.size(),false);// 关键帧地图点 匹配标记

    // 统计匹配点对的 方向差值  同一个匹配 方向相差不大
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    int nmatches = 0;

    // 将属于同一节点(特定层)的ORB特征进行匹配
    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it != f1end && f2it != f2end)
    {
//步骤1：分别取出属于同一node的ORB特征点(只有属于同一node，才有可能是匹配点)
        if(f1it->first == f2it->first)
        {
// 步骤2：遍历KF1中属于该node的特征点
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];
                // 取出KF1 中该特征对应的 地图点MapPoint
                MapPoint* pMP1 = vpMapPoints1[idx1];
                if(!pMP1)// 没有匹配的地图点跳过
                    continue;
                if(pMP1->isBad())// 是坏点 跳过
                    continue;

                const cv::Mat &d1 = Descriptors1.row(idx1);// 取出KF1中该特征对应的描述子

                int bestDist1=256;// 最好的距离（最小距离）
                int bestIdx2 =-1 ;
                int bestDist2=256;// 倒数第二好距离（倒数第二小距离）

// 步骤3：遍历KF2中属于该node的特征点，找到了最佳匹配点
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    const size_t idx2 = f2it->second[i2];

                    MapPoint* pMP2 = vpMapPoints2[idx2];
                    // 已经和KF1中某个点匹配过了,跳过,或者该特征点无匹配地图点,或者该地图点是坏点,跳过
                    if(vbMatched2[idx2] || !pMP2)
                        continue;

                    if(pMP2->isBad())
                        continue;
// 步骤4：求描述子的距离 保留最小和次小距离对应的 匹配点
                    const cv::Mat &d2 = Descriptors2.row(idx2);// 取出F中该特征对应的描述子
                    int dist = DescriptorDistance(d1,d2);
                    if(dist<bestDist1)// dist < bestDist1 < bestDist2，更新bestDist1 bestDist2
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdx2=idx2;// 对应KF2地图点下标
                    }
                    else if(dist<bestDist2)// bestDist1 < dist < bestDist2，更新bestDist2
                    {
                        bestDist2=dist;
                    }
                }
// 步骤5：根据阈值 和 角度投票剔除误匹配
                if(bestDist1<TH_LOW)
                {
                    // trick!
                    // 最佳匹配比次佳匹配明显要好，那么最佳匹配才真正靠谱
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        vpMatches12[idx1]=vpMapPoints2[bestIdx2];// 匹配到的对应KF2中的地图点
                        vbMatched2[bestIdx2]=true;// KF2 中的地图点 已经和 KF1中的某个地图点匹配

                        if(mbCheckOrientation)
                        {
                            // trick!
                            // angle：每个特征点在提取描述子时的旋转主方向角度，如果图像旋转了，这个角度将发生改变
                            // 所有的特征点的角度变化应该是一致的，通过直方图统计得到最准确的角度变化值
                            float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);//匹配点方向差 直方图
                        }
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    // 根据方向差一致性约束 剔除误匹配的点
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;
        // 统计直方图最高的三个bin保留，其他范围内的匹配点剔除。
        // 另外，若最高的比第二高的高10倍以上，则只保留最高的bin中的匹配点。
        // 若最高的比第 三高的高10倍以上，则 保留最高的和第二高bin中的匹配点。
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            // 如果特征点的旋转角度变化量属于这三个组，则保留 该匹配点对
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                // 将除了ind1 ind2 ind3以外的匹配点去掉
                vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                       vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo)
{    
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    //Compute epipole in second image
    cv::Mat Cw = pKF1->GetCameraCenter();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    cv::Mat C2 = R2w*Cw+t2w;
    const float invz = 1.0f/C2.at<float>(2);
    const float ex =pKF2->fx*C2.at<float>(0)*invz+pKF2->cx;
    const float ey =pKF2->fy*C2.at<float>(1)*invz+pKF2->cy;

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches=0;
    vector<bool> vbMatched2(pKF2->N,false);
    vector<int> vMatches12(pKF1->N,-1);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it!=f1end && f2it!=f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];
                
                MapPoint* pMP1 = pKF1->GetMapPoint(idx1);
                
                // If there is already a MapPoint skip
                if(pMP1)
                    continue;

                const bool bStereo1 = pKF1->mvuRight[idx1]>=0;

                if(bOnlyStereo)
                    if(!bStereo1)
                        continue;
                
                const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];
                
                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);
                
                int bestDist = TH_LOW;
                int bestIdx2 = -1;
                
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2];
                    
                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2);
                    
                    // If we have already matched or there is a MapPoint skip
                    if(vbMatched2[idx2] || pMP2)
                        continue;

                    const bool bStereo2 = pKF2->mvuRight[idx2]>=0;

                    if(bOnlyStereo)
                        if(!bStereo2)
                            continue;
                    
                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);
                    
                    const int dist = DescriptorDistance(d1,d2);
                    
                    if(dist>TH_LOW || dist>bestDist)
                        continue;

                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

                    if(!bStereo1 && !bStereo2)
                    {
                        const float distex = ex-kp2.pt.x;
                        const float distey = ey-kp2.pt.y;
                        if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
                            continue;
                    }

                    if(CheckDistEpipolarLine(kp1,kp2,F12,pKF2))
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }
                
                if(bestIdx2>=0)
                {
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];
                    vMatches12[idx1]=bestIdx2;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = kp1.angle-kp2.angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vMatches12[rotHist[i][j]]=-1;
                nmatches--;
            }
        }

    }

    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0)
            continue;
        vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
    }

    return nmatches;
}

/**
* @brief 将MapPoints投影到关键帧pKF中，并判断是否有重复的MapPoints
* 1.如果MapPoint能匹配关键帧的特征点，并且该特征点有对应的MapPoint，那么将两个MapPoint合并（选择观测数多的）
* 2.如果MapPoint能匹配关键帧的特征点，并且该特征点没有对应的MapPoint，那么为该特征点点添加地图点MapPoint
* @param  pKF         相邻关键帧
* @param  vpMapPoints 需要融合的 当前帧上的 MapPoints
* @param  th          搜索半径的因子
* @return             重复MapPoints的数量
*/
int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th)
{
    // 关键帧的旋转矩阵和平移矩阵 欧式变换
    cv::Mat Rcw = pKF->GetRotation();
    cv::Mat tcw = pKF->GetTranslation();

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;// 基线×f

    cv::Mat Ow = pKF->GetCameraCenter();// 关键帧的相机坐标中心点坐标
    int nFused=0;// 融合地图点的数量
    const int nMPs = vpMapPoints.size();// 需要融合的地图点数量
//步骤1：  遍历所有的MapPoints
    for(int i=0; i<nMPs; i++)
    {
        MapPoint* pMP = vpMapPoints[i];
//步骤2： 跳过不好的地图点 和 地图点被关键帧观测到，已经匹配好了，不用融合
        if(!pMP)
            continue;
        if(pMP->isBad() || pMP->IsInKeyFrame(pKF))
            continue;
// 步骤3： 将地图点投影在关键帧图像像素坐标上
        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc = Rcw*p3Dw + tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        const float invz = 1/p3Dc.at<float>(2);// 深度归一化因子
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;
        const float u = fx*x+cx;// 像素坐标
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        const float ur = u-bf*invz;//匹配点横坐标，深度相机和双目相机有
//步骤4：  判断距离是否在尺度协方差范围内
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        //地图点距离相机的距离，进而推断，在图像金字塔中可能的尺度，越远尺度小，越近尺度大
        const float dist3D = cv::norm(PO);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

//步骤5：观察视角必须小于60度  Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();
        if(PO.dot(Pn)<0.5*dist3D)
            continue;
        // 根据深度预测地图点在帧图像上的尺度，深度大尺度小，深度小尺度大
        int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

// 步骤6： 根据尺度确定搜索半径，进而在图像上确定候选关键点
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

//  步骤7：遍历候选关键点,计算与地图点描述子匹配距离，保留最近距离的匹配
        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();// 地图点描述子
        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            //  关键点的尺度 需要在 预测尺度 之上
            const cv::KeyPoint &kp = pKF->mvKeysUn[idx];// 关键帧 候选关键点
            const int &kpLevel= kp.octave;
            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;
// 步骤8：计算MapPoint投影的坐标与这个区域特征点的距离，如果偏差很大，直接跳过特征点匹配
            if(pKF->mvuRight[idx]>=0) // 深度/双目相机有右图像匹配点横坐标差值
            {
                // Check reprojection error in stereo
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float &kpr = pKF->mvuRight[idx];
                const float ex = u-kpx; // 横坐标差值
                const float ey = v-kpy; //纵坐标差值
                const float er = ur-kpr; // 右图像匹配点横坐标差值
                const float e2 = ex*ex+ey*ey+er*er;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                    continue;// 差值过大 直接跳过
            }
            else    //单目相机,无右图像 匹配点横坐标差值
            {
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float e2 = ex*ex+ey*ey;
                if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)// 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
                    continue;
            }
// 步骤9：计算地图点和关键帧特征点描述子之间的距离,选出最近距离的关键点
            const cv::Mat &dKF = pKF->mDescriptors.row(idx);
            const int dist = DescriptorDistance(dMP,dKF);
            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        // 找到了地图点MapPoint在该区域最佳匹配的特征点
        if(bestDist<=TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
// 步骤10： 如果MapPoint能匹配关键帧的特征点，并且该特征点有对应的MapPoint，那么将两个MapPoint合并（选择观测数多的）
            // 本身已经 匹配到 地图点
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                {
                    if(pMPinKF->Observations()>pMP->Observations())
                        pMP->Replace(pMPinKF);//原地图点用帧地图点替代
                    else
                        pMPinKF->Replace(pMP);
                }
            }
// 步骤11： 如果MapPoint能匹配关键帧的特征点，并且该特征点没有对应的MapPoint，那么为该特征点点添加地图点MapPoint
            //    关键帧特征点还没有匹配的地图点,把匹配到的地图点对应上去
            else
            {
                pMP->AddObservation(pKF,bestIdx);// pMP地图点观测到了帧pKF上第 bestIdx 个特征点
                pKF->AddMapPoint(pMP,bestIdx);// 帧的第 bestIdx 个特征点对应pMP地图点
            }
            nFused++;
        }
    }

    return nFused;
}

    /**
    * @brief 将MapPoints投影到 关键帧pKF 中，并判断是否有重复的MapPoints
    * Scw为世界坐标系到pKF机体坐标系的Sim3 相似变换变换 ，
    * 需要先将相似变换转换到欧式变换SE3 下  将世界坐标系下的vpPoints变换到机体坐标系
    * 1 地图点匹配到 帧 关键点 关键点有对应的地图点时， 用帧关键点对应的地图点 替换 原地图点
    * 2 地图点匹配到 帧 关键点 关键点无对应的地图点时，为该特征点 添加匹配到的地图点MapPoint
    * @param  pKF            相邻关键帧
    * @param  Scw            世界坐标系到pKF机体坐标系的Sim3 相似变换变换  [s*R t]
    * @param  vpPoints       需要融合的地图点 MapPoints
    * @param  th             搜索半径的因子
    * @param  vpReplacePoint
    * @return                重复MapPoints的数量
    */
int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
{
    // 相机内参数， Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // 相似变换Sim3 转换到 欧式变换SE3， Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));//相似变换里的 旋转矩阵的 相似尺度因子
    cv::Mat Rcw = sRcw/scw;// 欧式变换 里 的旋转矩阵
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;// 欧式变换 里 的 平移向量
    cv::Mat Ow = -Rcw.t()*tcw;//相机中心在 世界坐标系下的 坐标

    // Set of MapPoints already found in the KeyFrame
    // 关键帧已有的 匹配地图点
    const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();
    int nFused=0;// 融合计数
    const int nPoints = vpPoints.size();// 需要融合的 地图点 数量

    // For each candidate MapPoint project and match
// 步骤1： 遍历所有需要融合的 地图点 MapPoints
    for(int iMP=0; iMP<nPoints; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];
//步骤2： 跳过不好的地图点
        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;
//步骤3： 地图点投影到关键帧像素平面上，不在平面内的不考虑
        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();// 地图点世界坐标系坐标
        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;// 地图点在帧坐标系下的坐标

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)// 地图点在相机前方 深度不能为负值
            continue;

        // 投影到像素平面 Project into Image
        const float invz = 1.0/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))// 不在图像内 跳过
            continue;
//步骤4：  判断距离是否在尺度协方差范围内
        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        //地图点 距离相机的距离 进而推断 在图像金字塔中可能的尺度 越远尺度小 越近尺度大
        const float dist3D = cv::norm(PO);
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
//步骤5：观察视角 必须小于 60度  Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();
        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        // Compute predicted scale level
        // 根据深度预测地图点在帧图像上的尺度，深度大尺度小，深度小尺度大
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);
// 步骤6： 根据尺度确定搜索半径 进而在图像上确定 候选 关键点
        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);
        if(vIndices.empty())
            continue;

//  步骤7：遍历候选关键点  计算与地图点  描述子匹配 计算距离 保留最近距离的匹配
        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
        {
            const size_t idx = *vit;
            const int &kpLevel = pKF->mvKeysUn[idx].octave;
            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)//  关键点的尺度需要在预测尺度上
                continue;

// 步骤8：计算地图点和 关键帧 特征点 描述子之间的距离 选出最近距离的 关键点
            const cv::Mat &dKF = pKF->mDescriptors.row(idx);
            int dist = DescriptorDistance(dMP,dKF);
            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // 找到了地图点MapPoint在该区域最佳匹配的特征点
        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)//<50
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
// 步骤9： 如果MapPoint能匹配关键帧的特征点，并且该特征点有对应的MapPoint，
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                    vpReplacePoint[iMP] = pMPinKF;// 用关键点对应的地图点 替换 原地图点
            }
// 步骤10：  如果MapPoint能匹配关键帧的特征点，并且该特征点没有对应的MapPoint，那么为该特征点点添加地图点MapPoint
            else
            {
                pMP->AddObservation(pKF,bestIdx);// pMP地图点观测到了帧pKF上第bestIdx个特征点
                pKF->AddMapPoint(pMP,bestIdx);// 帧的第 bestIdx 个特征点对应pMP地图点
            }
            nFused++;
        }
    }

    return nFused;
}

    /**
    * @brief  通过Sim3变换，确定pKF1的特征点在pKF2中的大致区域，
    * 同理，确定pKF2的特征点在pKF1中的大致区域
    * 在该区域内通过描述子进行匹配捕获pKF1和pKF2之前漏匹配的特征点，
    * 更新vpMatches12（之前使用SearchByBoW进行特征点匹配时会有漏匹配）
    * @param pKF1          关键帧1
    * @param pKF2          关键帧2
    * @param vpMatches12   两帧原有匹配点  帧1 特征点 匹配到 帧2 的地图点
    * @param s12              帧2->帧1 相似变换 尺度
    * @param R12             帧2->帧1  欧式变换 旋转矩阵
    * @param t12              帧2->帧1 欧式变换 平移向量
    * @param th       		 搜索半径参数
    * @return                     成功匹配的数量
    */
int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12,
                             const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th)
{
// 步骤1：变量初始化-----------------------------------------------------
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // 世界坐标系到帧1的欧式变换 Camera 1 from world
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    //世界坐标系到帧2的欧式变换 Camera 2 from world
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    //相似变换、旋转矩阵、平移向量，Transformation between cameras
    cv::Mat sR12 = s12*R12;// 帧2->帧1 相似变换旋转矩阵 = 帧2->帧1相似变换尺度 * 帧2->帧1欧式变换旋转矩阵
    cv::Mat sR21 = (1.0/s12)*R12.t();;// 帧1->帧2相似变换旋转矩阵 = 帧1->帧2相似变换尺度 * 帧1->帧2欧式变换旋转矩阵
    cv::Mat t21 = -sR21*t12;// 帧1->帧2相似变换 平移向量

    // 帧1地图点数量  关键点数量
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size();

    // 帧2地图点数量  关键点数量
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();

    // 来源于两帧先前已有的匹配
    vector<bool> vbAlreadyMatched1(N1,false);// 帧1在帧2中是否有匹配
    vector<bool> vbAlreadyMatched2(N2,false);// 帧2在帧1中是否有匹配

// 步骤2：用vpMatches12更新 已有的匹配 vbAlreadyMatched1和vbAlreadyMatched2------------------------------------
    for(int i=0; i<N1; i++)
    {
        MapPoint* pMP = vpMatches12[i];// 帧1特征点匹配到帧2的地图点
        if(pMP)
        {
            vbAlreadyMatched1[i]=true;//  帧1特征点已经有匹配到的 地图点了
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);//  帧2的地图点在帧2中对应的下标
            if(idx2>=0 && idx2<N2)// 在 帧2特征点个数范围内的话
                vbAlreadyMatched2[idx2]=true;
        }
    }

    // 新寻找的匹配
    vector<int> vnMatch1(N1,-1);
    vector<int> vnMatch2(N2,-1);
// 步骤3：通过Sim变换，确定pKF1的地图点在pKF2帧图像中的大致区域，
    //  在该区域内通过描述子进行匹配捕获pKF1和pKF2之前漏匹配的特征点，更新vpMatches12
    // （之前使用SearchByBoW进行特征点匹配时会有漏匹配）
    // 每一个帧1中的地图点 投影到 帧2 上
    // Transform from KF1 to KF2 and search
    for(int i1=0; i1<N1; i1++)
    {
        MapPoint* pMP = vpMapPoints1[i1];

        //步骤3.1： 跳过已有的匹配 和 不存在的点 以及坏点
        if(!pMP || vbAlreadyMatched1[i1])
            continue;

        if(pMP->isBad())
            continue;

        //                                       SE3						Sim3
        //步骤3.2： 帧1地图点(世界坐标系)-------> 帧1地图点(帧1坐标系)-------> 帧1地图点(帧2坐标系)---->帧2像素坐标系下
        // 帧1  pKF1 地图点在世界坐标系中的点坐标
        cv::Mat p3Dw = pMP->GetWorldPos();// 帧1地图点(世界坐标系)
        cv::Mat p3Dc1 = R1w*p3Dw + t1w;// 帧1地图点(帧1坐标系)
        cv::Mat p3Dc2 = sR21*p3Dc1 + t21;// 帧1地图点(帧2坐标系)

        // Depth must be positive
        if(p3Dc2.at<float>(2)<0.0)// 深度值必须为正 相机前方
            continue;

        const float invz = 1.0/p3Dc2.at<float>(2);
        const float x = p3Dc2.at<float>(0)*invz;// 投影到帧2像素平面上
        const float y = p3Dc2.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF2->IsInImage(u,v))// 坐标必须在 图像平面尺寸内
            continue;

        //步骤3.3：  判断帧1地图点距帧2的距离 是否在尺度协方差范围内
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc2);

        // Depth must be inside the scale invariance region
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // 步骤3.4： 根据深度确定尺度,再根据尺度确定搜索半径,进而在图像上确定候选关键点
        // Compute predicted octave 根据深度预测地图点在帧2图像上的尺度, 深度大尺度小; 深度小尺度大
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);// 尺度也就是在金字塔哪一层
        const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];// 再根据尺度确定搜索半径
        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);//进而在图像上确定候选关键点

        if(vIndices.empty())
            continue;

        // 步骤3.5：遍历候选关键点, 计算与地图点描述子计算距离,保留最近距离的匹配
        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();// 帧1地图点描述子

        int bestDist = INT_MAX;
        int bestIdx = -1;
        // 遍历搜索 帧2区域内的所有特征点，与帧1地图点pMP进行描述子匹配
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];// 帧2候选区域内的 关键点
            //  关键点的尺度需要在预测尺度nPredictedLevel上
            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);// 帧2 关键点描述子
            const int dist = DescriptorDistance(dMP,dKF);// 帧1 地图点描述子 和 帧2关键点描述子距离

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)// <=100
        {
            vnMatch1[i1]=bestIdx;// 帧1地图点匹配到帧2的关键点(也对应一个地图点)
        }
    }
// 步骤4：通过Sim变换，确定pKF2的地图点在pKF1帧图像中的大致区域，
    //         在该区域内通过描述子进行匹配捕获pKF2和pKF1之前漏匹配的特征点，更新vpMatches12
    //         （之前使用SearchByBoW进行特征点匹配时会有漏匹配）
    // 每一个帧2中的地图点 投影到 帧1 上
    for(int i2=0; i2<N2; i2++)
    {
        MapPoint* pMP = vpMapPoints2[i2];// 帧2 关键点匹配的 地图点
        //步骤4.1： 跳过已有的匹配 和 不存在的点 以及坏点
        if(!pMP || vbAlreadyMatched2[i2])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc2 = R2w*p3Dw + t2w;
        cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

        // Depth must be positive
        if(p3Dc1.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc1.at<float>(2);
        const float x = p3Dc1.at<float>(0)*invz;
        const float y = p3Dc1.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF1->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc1);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;
        }
    }

    // Check agreement
// 步骤5 检查两者的匹配 是否对应起来
    int nFound = 0;
    for(int i1=0; i1<N1; i1++)
    {
        int idx2 = vnMatch1[i1];// 帧1地图点匹配到的 帧2的关键点(也对应一个地图点)下标
        if(idx2>=0)
        {
            int idx1 = vnMatch2[idx2];// 帧2关键点对应的帧1地图点下标
            if(idx1==i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];// 更新帧1在帧2中匹配的地图点
                nFound++;
            }
        }
    }

    return nFound;
}

    // b. 匹配上一帧的地图点，即前后两帧匹配，用于TrackWithMotionModel
    //运动模型（Tracking with motion model）跟踪   速率较快  假设物体处于匀速运动
    // 用 上一帧的位姿和速度来估计当前帧的位姿使用的函数为TrackWithMotionModel()。
    //这里匹配是通过投影来与上一帧看到的地图点匹配，使用的是matcher.SearchByProjection()。
/**
 * @brief 通过投影，对上一帧的特征点(地图点)进行跟踪
 * 运动跟踪模式
 * 上一帧中包含了MapPoints，对这些MapPoints进行跟踪tracking，由此增加当前帧的MapPoints \n
 * 1. 将上一帧的MapPoints投影到当前帧(根据速度模型可以估计当前帧的Tcw)
 * 2. 在投影点附近根据描述子距离选取匹配，以及最终的方向投票机制进行剔除
 * @param  CurrentFrame 当前帧
 * @param  LastFrame       上一帧
 * @param  th                      搜索半径参数
 * @param  bMono             是否为单目
 * @return                           成功匹配的数量
 * @see SearchByBoW()
 */
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
{
    int nmatches = 0;

// 步骤1：变量初始化----------------------------------------------------------
    // Rotation Histogram (to check rotation consistency)
    // 匹配点观测方向差 直方图统计用来筛选最好的匹配
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    // 当前帧 旋转 平移矩阵
    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    const cv::Mat twc = -Rcw.t()*tcw;// w-->c，在w下表达

    // 上一帧 旋转 平移矩阵
    const cv::Mat Rlw = LastFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tlw = LastFrame.mTcw.rowRange(0,3).col(3);
    //上一帧到当前帧的平移向量， 在上一帧下表达
    const cv::Mat tlc = Rlw*twc+tlw;

    // 判断前进还是后退
    const bool bForward = tlc.at<float>(2)>CurrentFrame.mb && !bMono;
    // 非单目情况，如果Z>0且大于基线，则表示前进
    const bool bBackward = -tlc.at<float>(2)>CurrentFrame.mb && !bMono;
    // 非单目情况，如果Z<0,且绝对值大于基线，则表示后退

// 步骤2：遍历上一帧所有的关键点(对应地图点)-------------------------------------------
    for(int i=0; i<LastFrame.N; i++)
    {
        MapPoint* pMP = LastFrame.mvpMapPoints[i];

        if(pMP)
        {
            if(!LastFrame.mvbOutlier[i]) // 该地图点存在，且是内点
            {
// 步骤3： 上一帧地图点投影到当前帧像素平面上-----------------------------------------
                cv::Mat x3Dw = pMP->GetWorldPos();// 上一帧地图点（世界坐标系下）
                cv::Mat x3Dc = Rcw*x3Dw+tcw;//上一帧地图点（当前帧坐标系下）

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);// 深度>0 逆深度>0

                if(invzc<0)
                    continue;

                float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;
                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)// 需要在 图像尺寸内
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

// 步骤4： 在当前帧上确定候选点-----------------------------------------------------
                // NOTE 尺度越大,图像越小
                // 以下可以这么理解，例如一个有一定面积的圆点，在某个尺度n下它是一个特征点
                // 当前进时，圆点的面积增大，在某个尺度m下它是一个特征点，由于面积增大，则需要在更高的尺度下才能检测出来
                // 因此m>=n，对应前进的情况，nCurOctave>=nLastOctave。后退的情况可以类推
                int nLastOctave = LastFrame.mvKeys[i].octave;//  上一帧地图点对应特征点所处的尺度(金字塔层数)

                // Search in a window. Size depends on scale
                float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];//尺度越大，搜索范围越大

                vector<size_t> vIndices2;// 当前帧上投影点附近的 候选点

                if(bForward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave);
                else if(bBackward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, 0, nLastOctave);
                else // 没怎么运动，在上一帧尺度附近搜索;单目也走这个分支
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave-1, nLastOctave+1);

                if(vIndices2.empty())
                    continue;

// 步骤5：遍历候选关键点 ，计算与地图点描述子匹配，并计算距离，保留最近距离的匹配
                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                {
                    const size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2]) // 如果当前帧有地图点
                        if(CurrentFrame.mvpMapPoints[i2]->Observations()>0)//该对应地图点也有观测帧 则跳过
                            continue;//跳过不用再匹配地图点

                    // 双目和rgbd的情况，需要保证右图的点也在搜索半径以内
                    if(CurrentFrame.mvuRight[i2]>0)
                    {
                        const float ur = u - CurrentFrame.mbf*invzc;
                        const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                        if(er>radius)
                            continue;
                    }

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2); // 当前帧关键点描述子
                    const int dist = DescriptorDistance(dMP,d); // 描述子匹配距离

                    if(dist<bestDist)
                    {
                        bestDist=dist;//最短的距离
                        bestIdx2=i2;// 对应的当前帧关键点下标
                    }
                }

                if(bestDist<=TH_HIGH) // 最短距离小于 <100
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;// 为当前帧关键点匹配上一帧的地图点
                    nmatches++;

                    // 匹配点 观测方向差 一致性检测
                    if(mbCheckOrientation)
                    {
                        // 上一帧地图点的观测方向  -  当前帧特征点的观测方向
                        float rot = LastFrame.mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);//统计到对应的 方向直方图上
                    }
                }
            }
        }
    }

// 步骤6：根据方向差一致性约束 剔除误匹配的点
    //Apply rotation consistency
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        // 统计直方图最高的三个bin保留，其他范围内的匹配点剔除。
        // 另外，若最高的比第二高的高10倍以上，则只保留最高的bin中的匹配点。
        // 若最高的比第 三高的高10倍以上，则 保留最高的和第二高bin中的匹配点。
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

/**
     在关键帧地图点对应的描述子和当前帧关键点描述子匹配后的匹配点数少
     把关键帧地图点，通过当前帧个世界的变换关系，转换到当前帧坐标系下
     再根据相机内参数K投影到当前帧的像素坐标系下
     根据像素坐标所处的格子区域和估算的金字塔层级信息，得到和地图点匹配的当前帧候选特征点
     计算匹配距离

     1. 获取pKF对应的地图点vpMPs，遍历
         (1). 若该点为NULL、isBad或者在SearchByBow中已经匹配上（Relocalization中首先会通过SearchByBow匹配一次），抛弃；
     2. 通过当前帧的位姿，将世界坐标系下的地图点坐标转换为当前帧坐标系（相机坐标系）下的坐标
         (2). 投影点(u,v)不在畸变矫正过的图像范围内，地图点的距离dist3D不在地图点的可观测距离内（根据地图点对应的金字塔层数，
               也就是提取特征的neighbourhood尺寸），抛弃
     3. 通过地图点的距离dist3D，预测特征对应金字塔层nPredictedLevel，并获取搜索window大小（th*scale），在以上约束的范围内，
        搜索得到候选匹配点集合向量vIndices2
         const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);
     4. 计算地图点的描述子和候选匹配点描述子距离，获得最近距离的最佳匹配，但是也要满足距离<ORBdist。
     5. 最后，还需要通过直方图验证描述子的方向是否匹配
 */
/**
 * @brief 通过投影，对上一参考关键帧的特征点(地图点)进行跟踪
 * 重定位模式中的 跟踪关键帧模式    重定位中先通过 SearchByBow 在关键帧数据库中找到候选关键帧，再与每一个参考关键帧匹配，找的匹配效果最好的，完成定位
 * 上一参考关键帧中包含了MapPoints，对这些MapPoints进行跟踪tracking，由此增加当前帧的MapPoints \n
 * 1. 将上一参考关键帧的MapPoints投影到当前帧
 * 2. 在投影点附近根据描述子距离选取匹配，以及最终的方向投票机制进行剔除
 * @param  CurrentFrame   当前帧
 * @param  pKF                    上一帧参考关键帧
 * @param  sAlreadyFound 当前帧关键点匹配到地图点的情况
 * @param  th                       搜索半径参数
 * @param  ORBdist             匹配距离阈值
 * @return                             成功匹配的数量
 * @see SearchByBoW()
 */
    int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist)
{
    int nmatches = 0;
    //  当前帧旋转平移矩阵向量，相机坐标点
    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    const cv::Mat Ow = -Rcw.t()*tcw;

    // Rotation Histogram (to check rotation consistency)
    // 匹配点对观测方向一致性检测
    // 匹配点对观测方向差值 方向直方图
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

// 步骤1：获取关键帧pKF对应的地图点vpMPs，遍历
    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();// 所有关键帧中的地图点
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)// 获取关键帧对应的地图点vpMPs，遍历
    {
        MapPoint* pMP = vpMPs[i];//关键帧中的地图点
        if(pMP)// 地图点存在
        {
            // 1). 若该点为NULL、isBad或者
            // 在SearchByBow中已经匹配上（Relocalization中首先会通过SearchByBow匹配一次），抛弃；
            if(!pMP->isBad() && !sAlreadyFound.count(pMP))
            {
                //Project
// 步骤2：关键帧 对应的有效地图点投影到 当前帧 像素平面上 查看是否在视野内
                cv::Mat x3Dw = pMP->GetWorldPos();// 关键帧地图点在世界坐标系下的坐标
                cv::Mat x3Dc = Rcw*x3Dw+tcw;// 关键帧地图点在 当前帧坐标系（相机坐标系）下的坐标
                // 得到归一化相机平面上的点
                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);
                // 有相机内参数 得到在像素平面上的投影点(u,v) 像素坐标
                const float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                const float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;
                //  投影点(u,v)不在畸变矫正过的图像范围内 抛弃
                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

                // Compute predicted scale level
// 步骤2： 地图点的距离dist3D不在地图点的可观测距离内（根据地图点对应的金字塔层数，
                //也就是提取特征的neighbourhood尺寸），抛弃
                cv::Mat PO = x3Dw-Ow;// 关键帧地图点到 当前帧相机中的的相对坐标
                float dist3D = cv::norm(PO);//关键帧地图点 距离当前帧相机中心的距离

                const float maxDistance = pMP->GetMaxDistanceInvariance();
                //地图点的可观测距离内（根据地图点对应的金字塔层数，也就是提取特征的neighbourhood尺寸）
                const float minDistance = pMP->GetMinDistanceInvariance();
                // 地图点的距离dist3D不在地图点的可观测距离内
                // Depth must be inside the scale pyramid of the image
                if(dist3D<minDistance || dist3D>maxDistance)
                    continue;

// 步骤3：通过地图点的距离dist3D，预测特征对应金字塔层nPredictedLevel，得到搜索半径，得到候选匹配点
                // 并获取搜索window大小（th*scale），在以上约束的范围内，
                // 搜索得到候选匹配点集合向量vIndices2
                int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);//通过地图点的距离dist3D，预测特征对应金字塔层nPredictedLevel
                // Search in a window 并获取搜索window大小（th*scale），
                const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];
                // 在以上约束的范围内，搜索得到候选匹配点集合向量vIndices2
                // 对于特征点格子内，图像金字塔的相应层上的候选特征点
                const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);
                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();//关键帧地图点的描述子
                int bestDist = 256;
                int bestIdx2 = -1;
// 步骤4：计算地图点的描述子和候选匹配点描述子距离，获得最近距离的最佳匹配，但是也要满足距离<ORBdist。
                for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)//每一个候选匹配点
                {
                    const size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2])//当前帧每一个候选匹配点 已经匹配到了 地图点 跳过
                        continue;

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);// 候选匹配点描述子
                    const int dist = DescriptorDistance(dMP,d);//  计算地图点的描述子和候选匹配点描述子距离
                    if(dist<bestDist)// 获得最近距离的最佳匹配，
                    {
                        bestDist=dist;//最短的距离
                        bestIdx2=i2;//对应的 当前帧 关键点下标
                    }
                }
// 步骤5：最短距离阈值检测  要满足最短距离距离<ORBdist。
                if(bestDist<=ORBdist)//但是也要满足距离<ORBdist 100
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;// 为当前帧生成和关键帧匹配上的地图点
                    nmatches++;
                    if(mbCheckOrientation)//  最后，还需要通过直方图验证描述子的方向是否匹配
                    {
                        float rot = pKF->mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;//匹配点观测方向差
                        //将关键帧与当前帧匹配点的观测方向angle相减
                        // 得到rot（0<=rot<360），放入一个直方图中
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);

                        // 对于每一对匹配点的角度差，均可以放入一个bin的范围内（360/HISTO_LENGTH）
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);// 方向直方图
                    }
                }

            }
        }
    }

// 步骤6：匹配点对 观测方向一致性 检测
    // 其中角度直方图是用来剔除不满足两帧之间角度旋转的外点的，也就是所谓的旋转一致性检测
    if(mbCheckOrientation)//  最后，还需要通过直方图验证描述子的方向是否匹配
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;
        // 统计直方图最高的三个bin保留，其他范围内的匹配点剔除。
        // 另外，若最高的比第二高的高10倍以上，则只保留最高的bin中的匹配点。
        // 若最高的比第 三高的高10倍以上，则 保留最高的和第二高bin中的匹配点。
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)//最高的三个bin保留
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;// 其他范围内的匹配点剔除
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

// 取出直方图中值最大的三个index
void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
/**
 * @brief     二进制向量之间 相似度匹配距离
 * @param  a  二进制向量
 * @param  b  二进制向量
 * @return
 */
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)//只计算了前八个 二进制位 的差异
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

} //namespace ORB_SLAM
