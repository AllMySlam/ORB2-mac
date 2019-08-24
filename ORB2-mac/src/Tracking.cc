/** 跟踪线程 深度 双目初始化位姿 运动模型 关键帧模式 重定位 局部地图跟踪 关键帧
*This file is part of ORB-SLAM2.
*
* mpMap就是我们整个位姿与地图（可以想象成ORB-SLAM运行时的那个界面世界），
* MapPoint和KeyFrame都被包含在这个mpMap中。
* 因此创建这三者对象（地图，地图点，关键帧）时，
* 三者之间的关系在构造函数中不能缺少。
*
* 另外，由于一个关键帧提取出的特征点对应一个地图点集，
* 因此需要记下每个地图点的在该帧中的编号；
*
* 同理，一个地图点会被多帧关键帧观测到，
* 也需要几下每个关键帧在该点中的编号。
*
* 地图点，还需要完成两个运算，第一个是在观测到该地图点的多个特征点中（对应多个关键帧），
* 挑选出区分度最高的描述子，作为这个地图点的描述子；
* pNewMP->ComputeDistinctiveDescriptors();
*
* 第二个是更新该地图点平均观测方向与观测距离的范围，这些都是为了后面做描述子融合做准备。
pNewMP->UpdateNormalAndDepth();

*
* 跟踪流程：
* 每一帧图像 Frame --->
* 提取ORB关键点特征 ----->
* 根据上一帧进行位置估计计算R t (或者通过全局重定位初始化位置) ------>
* 跟踪局部地图，优化位姿 ------->
* 是否加入 关键帧
*
* Tracking线程
* 帧 Frame
* 1】初始化
*       单目初始化 MonocularInitialization()
*       双目初始化 StereoInitialization
*
* 2】相机位姿跟踪P
*       同时跟踪和定位 同时跟踪与定位，不插入关键帧，局部建图 不工作
*       跟踪和定位分离 mbOnlyTracking(false)
        位姿跟踪 TrackWithMotionModel()  TrackReferenceKeyFrame()  重定位 Relocalization()
*
 a 运动模型（Tracking with motion model）跟踪   速率较快  假设物体处于匀速运动
      用 上一帧的位姿和速度来估计当前帧的位姿使用的函数为TrackWithMotionModel()。
      这里匹配是通过投影来与上一帧看到的地图点匹配，使用的是
      matcher.SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, ...)。

 b 关键帧模式      TrackReferenceKeyFrame()
     当使用运动模式匹配到的特征点数较少时，就会选用关键帧模式。即尝试和最近一个关键帧去做匹配。
     为了快速匹配，本文利用了bag of words（BoW）来加速。
     首先，计算当前帧的BoW，并设定初始位姿为上一帧的位姿；
     其次，根据位姿和BoW词典来寻找特征匹配，使用函数matcher.SearchByBoW(KeyFrame *pKF, Frame &F, ...)；
     匹配到的是参考关键帧中的地图点。
     最后，利用匹配的特征优化位姿。

c 通过全局重定位来初始化位姿估计 Relocalization()
    假如使用上面的方法，当前帧与最近邻关键帧的匹配也失败了，
    那么意味着需要重新定位才能继续跟踪。
    重定位的入口如下： bOK = Relocalization();
    此时，只有去和所有关键帧匹配，看能否找到合适的位置。
    首先，计算当前帧的BOW向量，在关键帧词典数据库中选取若干关键帧作为候选。
         使用函数如下：vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);
    其次，寻找有足够多的特征点匹配的关键帧；最后，利用RANSAC迭代，然后使用PnP算法求解位姿。这一部分也在Tracking::Relocalization() 里


 * 3】局部地图跟踪
*       更新局部地图 UpdateLocalMap() 更新关键帧和 更新地图点  UpdateLocalKeyFrames()   UpdateLocalPoints
*       搜索地图点  获得局部地图与当前帧的匹配
*       优化位姿    最小化重投影误差  3D点-2D点对  si * pi = K * T * Pi = K * exp(f) * Pi
*
* 4】是否生成关键帧
*       加入的条件：
*       很长时间没有插入关键帧
*       局部地图空闲
*       跟踪快要跟丢
*       跟踪地图 的 MapPoints 地图点 比例比较少
*
* 5】生成关键帧
*       KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB)
*       对于双目 或 RGBD摄像头构造一些 MapPoints，为MapPoints添加属性
*
* 进入LocalMapping线程
*
*
*/


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>
#include <unistd.h>

using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

// 输入左目RGB或RGBA图像
// 1、将图像转为mImGray并初始化mCurrentFrame
// 2、进行tracking过程
// 输出世界坐标系到该帧相机坐标系的变换矩阵
cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;

    // 步骤1：将RGB或RGBA图像转为灰度图像
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    // 步骤2：构造Frame
    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)// 没有成功初始化的前一个状态就是NO_IMAGES_YET
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);//提取2倍的特征点
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    // 步骤3：跟踪
    Track();

    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track()
{
    // track包含两部分：估计运动、跟踪局部地图

    // mState为tracking的状态机
    // SYSTME_NOT_READY, NO_IMAGE_YET, NOT_INITIALIZED, OK, LOST
    // 如果图像复位过、或者第一次运行，则为 NO_IMAGE_YET 状态
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    // mLastProcessedState存储了Tracking最新的状态，用于FrameDrawer中的绘制
    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    // 步骤1：初始化
    if(mState==NOT_INITIALIZED)
    {
        // 1. 单目/双目/RGBD初始化 得到第一帧下看到的3d点
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            // 连续两帧特征点个数大于100个 且两帧 关键点orb特征匹配点对数 大于100个
            // 初始帧 [I  0] 第二帧 基础矩阵/单应恢复 [R t] 全局优化,同时得到对应的3D点
            // 创建地图,使用最小化重投影误差BA进行地图优化优化位姿和地图点
            // 深度距离中值倒数,归一化第二帧位姿的平移向量和地图点的三轴坐标
            // 显示更新
            MonocularInitialization();
        // 2. 可视化显示当前帧位姿
        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    // 步骤2：后面帧的跟踪 ==========================================================================================================
        // 1. 跟踪上一帧得到一个对位姿的初始估计.
        // 系统已经初始化(地图中已经有3d点) 跟踪上一帧特征点对，计算相机移动位姿----
        // 2. 跟踪局部地图, 图优化对位姿进行精细化调整
        // 3. 跟踪失败后的处理（两两跟踪失败 or 局部地图跟踪失败）
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        // 在viewer中有个开关 menuLocalizationMode ，有它控制是否 ActivateLocalizationMode ，并最终管控mbOnlyTracking
        // mbOnlyTracking等于false表示正常VO模式（有地图更新），mbOnlyTracking等于true表示用户手动选择定位模式
        // A1.跟踪上一帧=============================================================================================================
        // 1. 跟踪 + 建图 + 重定位（跟踪丢失后进行重定位）==============================================
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            // A. 正常初始化成功==============================================================================
            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                // 检查并更新上一帧被替换的MapPoints
                // 更新Fuse函数和 SearchAndFuse 函数替换的 MapPoints
                CheckReplacedInLastFrame();// 最后一帧 地图点 是否有替换点 有替换点的则进行替换

                // a. 跟踪参考帧模式 移动速度小========================================
                // 没有移动 跟踪参考关键帧(运动模型是空的)  或 刚完成重定位
                // mCurrentFrame.mnId < mnLastRelocFrameId+2这个判断不应该有
                // 应该只要mVelocity不为空，就优先选择TrackWithMotionModel
                // mnLastRelocFrameId 上一次重定位的那一帧
                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    // 将上一帧的位姿作为当前帧的初始位姿
                    // 通过BoW的方式在参考帧中找当前帧特征点的匹配点
                    // 优化每个特征点都对应3D点重投影误差即可得到位姿
                    bOK = TrackReferenceKeyFrame();// 跟踪参考关键帧中的地图点大于10个返回真
                }
                // b. 有移动 先进行移动 跟踪 模式=======================================
                else
                {
                    // 根据恒速模型设定当前帧的初始位姿
                    // 通过投影的方式，在上一帧参考帧中找当前帧特征点的匹配点
                    // 优化每个特征点所对应3D点的投影误差，即可得到位姿
                    bOK = TrackWithMotionModel();// 移动跟踪模式, 跟踪上一帧
                    if(!bOK)//没成功 ，则尝试进行跟踪参考帧模式
                        bOK = TrackReferenceKeyFrame();
                        // 不能根据固定运动速度模型预测当前帧的位姿态，通过bow加速匹配（SearchByBow）
                        // 最后通过优化得到优化后的位姿
                }
            }
            // B. 更丢了，重定位模式===========================================================================
            else
            {
                bOK = Relocalization();//重定位，BOW搜索，PnP 3d-2d匹配 求解位姿
            }
        }
        // 2. 已经有地图的情况下，则进行 跟踪 + 重定位（跟踪丢失后进行重定位）================================================
        else
        {
            // Localization Mode: Local Mapping is deactivated
            // A.跟踪丢失 ======================================================================================
            if(mState==LOST)
            {
                bOK = Relocalization();//重定位  BOW搜索，PnP 3d-2d匹配 求解位姿
            }
            else
            {
            // B. 正常跟踪
                // mbVO 是 mbOnlyTracking 为true时，才有的一个变量
                // mbVO 为0表示此帧匹配了很多的MapPoints，跟踪很正常，
                // mbVO 为1表明此帧匹配了很少的MapPoints，少于10个，要跪的节奏
                if(!mbVO)
                {
                    // a. 上一帧跟踪的点足够多=============================================
                    // 1. 移动跟踪模式, 如果失败，尝试使用跟踪参考帧模式====
                    if(!mVelocity.empty()) // 在移动
                    {
                        bOK = TrackWithMotionModel();// 恒速跟踪上一帧 模型
                    }
                    // 2. 使用跟踪参考帧模式===============================
                    else //未移动
                    {
                        bOK = TrackReferenceKeyFrame(); // 跟踪 参考帧
                    }
                }
                // b. 上一帧跟踪的点比较少(到了无纹理区域等)，要跪的节奏，既做跟踪又做定位========
                else // mbVO为1，则表明此帧匹配了很少的3D map点，少于10个，要跪的节奏，既做运动跟踪又做定位
                {
                    // 使用运动跟踪和重定位模式计算两个位姿，如果重定位成功，使用重定位得到的位姿
                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;// 视觉里程计跟踪得到的位姿结果
                    if(!mVelocity.empty()) // 有速度运动跟踪模式
                    {
                        bOKMM = TrackWithMotionModel();       // 运动跟踪模式跟踪上一帧 结果

                        vpMPsMM = mCurrentFrame.mvpMapPoints; // 地图点
                        vbOutMM = mCurrentFrame.mvbOutlier;   // 外点
                        TcwMM = mCurrentFrame.mTcw.clone();   // 保存视觉里程计位姿结果
                    }
                    bOKReloc = Relocalization(); // 重定位模式
                    // 1.重定位没有成功，但运动跟踪成功,使用跟踪的结果===================================
                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);           // 把帧的位置设置为视觉里程计位姿结果
                        mCurrentFrame.mvpMapPoints = vpMPsMM;   // 当前帧看到的地图点
                        mCurrentFrame.mvbOutlier = vbOutMM;     // 外点

                        if(mbVO)
                        {
                            // 这段代码是不是有点多余？应该放到TrackLocalMap函数中统一做
                            // 更新当前帧的MapPoints被观测程度
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    // 2. 重定位模式 成功=================================================
                    else if(bOKReloc) // 只要重定位成功整个跟踪过程正常进行（定位与跟踪，更相信重定位）
                    {
                        mbVO = false; //重定位成功
                    }

                    bOK = bOKReloc || bOKMM; // 运动跟踪/重定位 成功标志
                }
            }
        }

        // A2. 局部地图跟踪=======================================================================================================
            // 通过之前的计算，已经得到一个对位姿的初始估计，我们就能通过投影，
            // 从已经生成的地图点中找到更多的对应关系，来精确结果
            // 三种模式的初始跟踪之后进行局部地图的跟踪（
                /// 局部地图包括：
                /// 1 - K1个关键帧、K2个临近关键帧和参考关键帧
                /// 2 - 由这些关键帧观测到的MapPoints）
            // 局部地图点的描述子和当前帧特征点(还没有匹配到地图点的关键点)进行描述子匹配
            // 图优化进行优化,利用当前帧的特征点的像素坐标和与其匹配的3D地图点,在其原位姿上进行优化
            // 匹配优化后,成功的点对数一般情况下大于30认为成功
            // 在刚进行过重定位的情况下,需要大于50认为成功
        mCurrentFrame.mpReferenceKF = mpReferenceKF;// 参考关键帧

        // A2.1：在帧间匹配得到初始的姿态后，现在对local map进行跟踪得到更多的匹配，并优化当前位姿
            /// local map:当前帧、当前帧的MapPoints、当前关键帧与其它关键帧共视关系
            // 在上面两两帧跟踪（恒速模型跟踪上一帧、跟踪参考帧），
            // 这里搜索局部关键帧后,搜集所有局部MapPoints，
            // 然后将局部MapPoints和当前帧进行投影匹配，得到更多匹配的MapPoints后进行Pose优化

        // 有建图线程
        if(!mbOnlyTracking)
        {
            if(bOK)
                bOK = TrackLocalMap();// 局部地图跟踪g20优化 ------------------
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO) // 重定位成功
                bOK = TrackLocalMap(); // 局部地图跟踪--------------------------
        }

        if(bOK)
            mState = OK;
        else
            mState=LOST;// 丢失

        // Update drawer  更新显示
        mpFrameDrawer->Update(this);

        // A2.2 局部地图跟踪成功, 根据运动模型，清除外点等，检查是否需要创建新的关键帧
        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            // a. 有运动，则更新运动模型 Update motion model 运动速度为前后两针的变换矩阵
            if(!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc;//运动速度为前后两帧的变换矩阵
            }
            else
                mVelocity = cv::Mat();// 无速度

            // 显示 当前相机位姿
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            // b. 清除UpdateLastFrame中为当前帧临时添加的MapPoints
            // 当前帧地图点的观测帧数量小于1的话，清掉相应的地图点
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];//当前帧匹配到的地图点
                if(pMP)//指针存在
                    if(pMP->Observations()<1)// 其观测帧 小于 1
                    {
                        // 排除UpdateLastFrame函数中为了跟踪增加的MapPoints
                        mCurrentFrame.mvbOutlier[i] = false; // 外点标志 0
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);// 清掉相应的地图点
                    }
            }

            // c. 清除临时的MapPoints,删除临时的地图点
                // 这些MapPoints在TrackWithMotionModel的UpdateLastFrame函数里生成（仅双目和rgbd）
                // b中只是在当前帧中将这些MapPoints剔除，这里从MapPoints数据库中删除
                // 这里生成的仅仅是为了提高双目或rgbd摄像头的帧间跟踪效果，用完以后就扔了，没有添加到地图中
                //  list<MapPoint*>::iterator
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;// 删除地图点对应的内存空间
            }
            mlpTemporalPoints.clear();

            // d. 判断是否需要新建关键帧
                // 最后一步是确定是否将当前帧定为关键帧，由于在Local Mapping中，
                // 会剔除冗余关键帧，所以我们要尽快插入新的关键帧，这样才能更鲁棒。
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();

            // e. 外点清除，检查外点，标记(不符合变换矩阵的点优化时更新)
            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // A3. 跟踪失败后的处理（两两跟踪失败 or 局部地图跟踪失败）======================================================
        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)// 关键帧数量过少（刚开始建图） 直接退出
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);//新建关键帧
    }

    // 步骤3: 返回跟踪得到的位姿 信息=======================================================================
    // 计算参考帧到当前帧的变换 Tcr = mTcw  * mTwr
    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else //跟踪丢失会造成位姿为空
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

/**
 * @brief 单目的地图初始化
 *
 * 并行地计算基础矩阵和单应性矩阵，选取其中一个模型，恢复出最开始两帧之间的相对姿态以及点云
 * 得到初始两帧的匹配、相对运动、初始MapPoints
 */
void Tracking::MonocularInitialization()
{
    // 如果单目初始器还没有被创建，则创建单目初始器
    if(!mpInitializer)
    {
        // Set Reference Frame
        // 单目初始帧的特征点数必须大于100
        if(mCurrentFrame.mvKeys.size()>100)
        {
            // 步骤1：得到用于初始化的第一帧，初始化需要两帧
            mInitialFrame = Frame(mCurrentFrame);
            // 记录最近的一帧
            mLastFrame = Frame(mCurrentFrame);
            // mvbPrevMatched最大的情况就是所有特征点都被跟踪上
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;

            // 由当前帧构造初始器 sigma:1.0 iterations:200
            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);// 跟踪初始化时前两帧之间的匹配

            return;
        }
    }
    else
    {
        // Try to initialize
        // 步骤2：如果当前帧特征点数大于100，则得到用于单目初始化的第二帧
        // 如果当前帧特征点太少，重新构造初始器
        // 因此只有连续两帧的特征点个数都大于100时，才能继续进行初始化过程
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        // 步骤3：在mInitialFrame与mCurrentFrame中找匹配的特征点对
        // mvbPrevMatched为前一帧的特征点，存储了mInitialFrame中哪些点将进行接下来的匹配
        // mvIniMatches存储mInitialFrame,mCurrentFrame之间匹配的特征点
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        // 步骤4：如果初始化的两帧之间的匹配点太少，重新初始化
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        // 步骤5：通过H模型或F模型进行单目初始化，得到两帧间相对运动、初始MapPoints
        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            // 步骤6：删除那些无法进行三角化的匹配点
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            // 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            // 由Rcw和tcw构造Tcw,并赋值给mTcw，mTcw为世界坐标系到该帧的变换矩阵
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            // 步骤6：将三角化得到的3D点包装成MapPoints
            // Initialize函数会得到mvIniP3D，
            // mvIniP3D是cv::Point3f类型的一个容器，是个存放3D点的临时变量，
            // CreateInitialMapMonocular将3D点包装成MapPoint类型存入KeyFrame和Map中
            CreateInitialMapMonocular();//创建初始地图,使用最小化重投影误差BA进行:地图优化、优化位姿 和 地图点
        }
    }
}

/**
* @brief CreateInitialMapMonocular
* 初始帧设置为世界坐标系原点,初始化后解出来的当前帧位姿T,最小化重投影误差BA,全局优化位姿T
* 为单目摄像头三角化生成MapPoints
*/
void Tracking::CreateInitialMapMonocular()
{
    //【1】创建关键帧 Create KeyFrames
    // 构建初始地图就是将这两关键帧以及对应的地图点加入地图（mpMap）中，需要分别构造关键帧以及地图点
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);// 初始关键帧 加入地图 加入关键帧数据库
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);//当前关键帧 第二帧

    //【2】计算帧 描述子 在 描述子词典 中的 线性表示向量
    //TODO qzc
    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    //【3】地图中插入关键帧 Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // 【4】创建地图点 关联到 关键帧 Create MapPoints and asscoiate to keyframes
    // 地图点中需要加入其一些属性：
    //1. 观测到该地图点的关键帧（对应的关键点）；
    //2. 该MapPoint的描述子；
    //3. 该MapPoint的平均观测方向和观测距离。
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)// 不好的匹配不要
            continue;

        // 【5】创建地图点 Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);// mvIniP3D 三角化得到的 3D点  vector 3d转化成 mat 3d
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        pKFini->AddMapPoint(pMP,i);// 初始帧 添加地图点
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);// 当前帧 添加地图点

        // 【6】地图点添加观测帧,参考帧和当前帧均可以观测到该地图点
        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        // 【7】 更新地图点的一些新的参数 描述子 观测方向 观测距离
        pMP->ComputeDistinctiveDescriptors();// 地图点 在 所有观测帧上的 最具有代表性的 描述子
        pMP->UpdateNormalAndDepth();// 该MapPoint的平均观测方向和观测距离。
        // 更新 地图点 相对各个观测帧相机中心单位化坐标
        // 更新 地图点 在参考帧下各个金字塔层级下的最小最大距离

        // 【8】当前帧 关联到地图点
        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        //【9】地图 添加地图点
        mpMap->AddMapPoint(pMP);
    }

    // 【10】更新关键帧的连接关系，被观测的次数
    //      还需要更新关键帧之间连接关系（以共视地图点的数量作为权重）：
    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;
    // 【11】 全局优化地图 BA最小化重投影误差
    Optimizer::GlobalBundleAdjustemnt(mpMap,20);// 对这两帧姿态进行全局优化重投影误差（LM）：
    // 注意这里使用的是全局优化，和回环检测调整后的大回环优化使用的是同一个函数。

    // 【12】设置深度中值为1
    // 需要归一化第一帧中地图点深度的中位数；
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);//  单目环境深度中值
    float invMedianDepth = 1.0f/medianDepth;

    // 【13】检测重置  如果深度<0 或者 这时发现 优化后 第二帧追踪到的地图点<100，也需要重新初始化。
    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // 【14】关键帧位姿 平移量尺度归一化
    // 否则，将深度中值作为单位一，归一化第二帧的位姿与所有的地图点。
    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();// 位姿
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;// 平移量归一化尺度
    pKFcur->SetPose(Tc2w);//设置新的位姿

    // 【15】地图点 尺度归一化 Scale points
    // 地图点 归一化尺度
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);//地图点尺度归一化
        }
    }

    // 【16】 对象更新
    // 局部地图插入关键帧
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);
    // 当前帧 更新位姿
    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;// 当前帧迭代到上一帧，为下一次迭代做准备
    mpLastKeyFrame = pKFcur;
    // 局部关键帧 局部地图点更新
    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;// 参考关键帧
    mCurrentFrame.mpReferenceKF = pKFcur;// 当前帧的 参考帧

    mLastFrame = Frame(mCurrentFrame);// 当前帧 迭代到上一帧  为下一次迭代做准备
    // 参考地图点
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    // 地图显示
    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());
    // 地图关键帧序列
    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();// 有替换点
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;// 进行替换
            }
        }
    }
}

// 跟踪参考帧  机器人没怎么移动
// 当前帧特征点描述子 和 参考关键帧帧中的地图点 的描述子 进行 匹配
// 保留方向直方图中最高的三个bin中 关键点 匹配的 地图点  匹配点对
// 采用 词带向量匹配
// 关键帧和 当前帧 均用 字典单词线性表示
// 对应单词的 描述子 肯定比较相近 取对应单词的描述子进行匹配可以加速匹配
// 和参考关键帧的地图点匹配  匹配点对数 需要大于15个
// 使用 图优化 根据地图点 和 帧对应的像素点  在初始位姿的基础上 优化位姿
// 同时剔除  外点
// 最终超过10个 匹配点 的 返回true 跟踪成功
/**
 * @brief 对参考关键帧的MapPoints进行跟踪
 *
 * 1. 计算当前帧的词包，将当前帧的特征点分到特定层的nodes上
 * 2. 对属于同一node的描述子进行匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return 如果匹配数大于10，返回true
 */
bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    return nmatchesMap>=10;
}

// 更新上一帧信息
// 更新上一帧位姿    =  参考帧到上一帧Tlr * 世界到参考帧Trw = 世界到上一帧的转转换Tlw
// 更新上一帧地图点
/**
 * @brief 双目或rgbd摄像头根据深度值为上一帧产生新的MapPoints
 *
 * 在双目和rgbd情况下，选取一些深度小一些的点（可靠一些） \n
 * 可以通过深度值产生一些新的MapPoints
 */
void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF; //世界到参考帧Trw
    cv::Mat Tlr = mlRelativeFramePoses.back(); //上一帧的参考帧 到 上一帧的变换 Tlr
    mLastFrame.SetPose(Tlr*pRef->GetPose());// 更新上一帧位姿  =  参考帧到上一帧Tlr * 世界到参考帧Trw = 世界到上一帧的转转换Tlw

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

// 移动模式跟踪  移动前后两帧  得到 变换矩阵
// 上一帧的地图点 反投影到当前帧图像像素坐标上  和 当前帧的 关键点落在 同一个 格子内的
// 做描述子匹配 搜索 可以加快匹配
/*
 使用匀速模型估计的位姿，将LastFrame中临时地图点投影到当前姿态，
 在投影点附近根据描述子距离进行匹配（需要>20对匹配，否则匀速模型跟踪失败，
 运动变化太大时会出现这种情况），然后以运动模型预测的位姿为初值，优化当前位姿，
 优化完成后再剔除外点，若剩余的匹配依然>=10对，
 则跟踪成功，否则跟踪失败，需要Relocalization：

  运动模型（Tracking with motion model）跟踪   速率较快  假设物体处于匀速运动
      用 上一帧的位姿和速度来估计当前帧的位姿使用的函数为TrackWithMotionModel()。
      这里匹配是通过投影来与上一帧看到的地图点匹配，使用的是
      matcher.SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, ...)。
 */
/**
 * @brief 根据匀速度模型对上一帧的MapPoints进行跟踪
 *
 * 1. 非单目情况，需要对上一帧产生一些新的MapPoints（临时）
 * 2. 将上一帧的MapPoints投影到当前帧的图像平面上，在投影的位置进行区域匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return 如果匹配数大于10，返回true
 * @see V-B Initial Pose Estimation From Previous Frame
 */
bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);// 匹配点匹配器 最小距离 < 0.9*次短距离 匹配成功

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    // 更新上一帧位姿    =  参考帧到上一帧Tlr * 世界到参考帧Trw = 世界到上一帧的转转换Tlw
    // 更新上一帧地图点
    UpdateLastFrame();

    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);//Tcl * Tlw = Tcw
    // 其中 mVelocity = mCurrentFrame.mTcw * LastTwc（Twl） = Tcl；运动速度为上一帧到当前帧的变换矩阵

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;// 搜索窗口
    else
        th=7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }    

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap>=10;
}

// 三种模式的初始跟踪之后进行局部地图的跟踪
// 局部地图点的描述子和当前帧特征点(还没有匹配到地图点的关键点),进行描述子匹配
// 图优化进行优化,利用当前帧的特征点的像素坐标和与其匹配的3D地图点,在其原位姿上进行优化
// 匹配优化后,成功的点对数一般情况下大于30认为成功
// 在刚进行过重定位的情况下需要大于50认为成功
/*
 以上两种仅仅完成了视觉里程计中的帧间跟踪，
 还需要进行局部地图的跟踪，提高精度：（这其实是Local Mapping线程中干的事情）
 局部地图跟踪TrackLocalMap()中需要
     首先对局部地图进行更新(UpdateLocalMap)，
     并且搜索局部地图点(SearchLocalPoint)。
     局部地图的更新又分为
         局部地图点(UpdateLocalPoints) 和
         局部关键帧(UpdateLocalKeyFrames)的更新.

 为了降低复杂度，这里只是在局部图中做投影。局部地图中与当前帧有相同点的关键帧序列成为K1，
 在covisibility graph中与K1相邻的称为K2。局部地图有一个参考关键帧Kref∈K1，
 它与当前帧具有最多共同看到的地图云点。针对K1, K2可见的每个地图云点，
 通过如下步骤，在当前帧中进行搜索:

（1）将地图点投影到当前帧上，如果超出图像范围，就将其舍弃；
（2）计算当前视线方向向量v与地图点云平均视线方向向量n的夹角，舍弃n·v < cos(60°)的点云；
（3）计算地图点到相机中心的距离d，认为[dmin, dmax]是尺度不变的区域，若d不在这个区域，就将其舍弃；
（4）计算图像的尺度因子，为d/dmin；
（5）将地图点的特征描述子D与还未匹配上的ORB特征进行比较，根据前面的尺度因子，找到最佳匹配。
  这样，相机位姿就能通过匹配所有地图点，最终被优化。
  
 */
/**
 * @brief 对Local Map的MapPoints进行跟踪
 *
 * 1. 更新局部地图，包括局部关键帧和关键点
 * 2. 对局部MapPoints进行投影匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return true if success
 * @see V-D track Local Map
 */
bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
/// 【1】首先对局部地图进行更新(UpdateLocalMap) 生成对应当前帧的 局部地图
    // 更新局部地图(与当前帧相关的帧和地图点) 用于局部地图点的跟踪 : 关键帧 + 地图点
    // 更新局部关键帧-------局部地图的一部分，共视化程度高的关键帧、子关键帧、父关键帧
    // 局部地图点的更新比较容易，完全根据局部关键帧来，所有局部关键帧的地图点就构成局部地图点
    UpdateLocalMap();
///【2】并且搜索局部地图点(SearchLocalPoint)
    // 局部地图点搜寻和当前帧关键点描述子的匹配，有匹配的加入到当前帧特征点对应的地图点中
    SearchLocalPoints();

/// 【3】优化帧位姿 Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    // 优化时会更新当前帧的位姿变换关系,同时更新地图点的内点/外点标记
    mnMatchesInliers = 0;

/// 【4】更新地图点状态 Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])//特征点找到地图点
        {
            if(!mCurrentFrame.mvbOutlier[i])//是内点符合变换关系
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();// 特征点找到地图点标志
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)// 外点,在双目下,清空匹配的地图点
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;//刚刚进行过重定位 则需要 匹配点对数大于 50 才认为 成功

    if(mnMatchesInliers<30)//正常情况下  找到的匹配点对数 大于 30 算成功
        return false;
    else
        return true;
}

/*
确定关键帧的标准如下：
（1）在上一个全局重定位后，又过了20帧；
（2）局部建图闲置，或在上一个关键帧插入后，又过了20帧；
（3)当前帧跟踪到大于50个点；
（4）当前帧跟踪到的比参考关键帧少90%。
*/
/**
 * @brief 断当前帧是否为关键帧
 * @return true if needed
 */
bool Tracking::NeedNewKeyFrame()
{
// 步骤1：如果用户在界面上选择重定位，那么将不插入关键帧
    // 由于插入关键帧过程中会生成MapPoint，因此用户选择重定位后地图上的点云和关键帧都不会再增加
    if(mbOnlyTracking) // 不建图不需要关键帧
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    // 建图线程 停止了
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();// 地图中的关键帧数量

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    // 刚刚重定位不久不需要插入关键帧 ，关键帧总数超过最大值也不需要插入关键帧
// 步骤2：判断是否距离上一次插入关键帧的时间太短
    // mCurrentFrame.mnId是当前帧的ID
    // mnLastRelocFrameId是最近一次重定位帧的ID
    // mMaxFrames等于图像输入的帧率  mMaxFrames = fps;
    // 如果关键帧比较少，则考虑插入关键帧
    // 或距离上一次重定位超过1s，则考虑插入关键帧
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

// 步骤3：得到参考关键帧跟踪到的MapPoints数量
    // 在UpdateLocalKeyFrames函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧
    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

// 步骤4：查询局部地图管理器是否繁忙
    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

// 步骤5：对于双目或RGBD摄像头，统计总的可以添加的MapPoints数量和跟踪到地图中的MapPoints数量
    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR) // 双目或rgbd
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

// 步骤6：决策是否需要插入关键帧
    // Thresholds 设定inlier阈值，和之前帧特征点匹配的inlier比例
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;// 关键帧只有一帧，那么插入关键帧的阈值设置很低

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    // 很长时间没有插入关键帧
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    // localMapper处于空闲状态
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    // 跟踪要跪的节奏，0.25和0.3是一个比较低的阈值
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    // 阈值比c1c要高，与之前参考帧（最近的一个关键帧）重复度不是太高
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                // 队列里不能阻塞太多关键帧
                // tracking插入关键帧不是直接插入，而且先插入到mlNewKeyFrames中，
                // 然后localmapper再逐个pop出来插入到mspKeyFrames
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

/**
 * @brief 创建新的关键帧
 *
 * 对于非单目的情况，同时创建新的MapPoints
 */
void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

// 步骤1：将当前帧构造成关键帧
    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

// 步骤2：将当前关键帧设置为当前帧的参考关键帧
    // 在UpdateLocalKeyFrames函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

// 步骤3：对于双目或rgbd摄像头，为当前帧生成新的MapPoints
    // 这段代码和UpdateLastFrame中的那一部分代码功能相同
    if(mSensor!=System::MONOCULAR)
    {
        // 根据Tcw计算mRcw、mtcw和mRwc、mOw
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        // 双目 / 深度
        // 步骤3.1：得到当前帧深度小于阈值的特征点
        // 创建新的MapPoint, depth < mThDepth
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            // 步骤3.2：按照深度从小到大排序
            sort(vDepthIdx.begin(),vDepthIdx.end());
            // 步骤3.3：将距离比较近的点包装成MapPoints
            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    // 这些添加属性的操作是每次创建MapPoint后都要做的
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }
                // 这里决定了双目和rgbd摄像头时地图点云的稠密程度
                // 但是仅仅为了让地图稠密直接改这些不太好，
                // 因为这些MapPoints会参与之后整个slam过程
                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

/**
 * @brief 对Local MapPoints进行跟踪
 * 搜索 在对应当前帧的局部地图内搜寻和当前帧地图点匹配点的局部地图点
 * 局部地图点搜寻和当前帧关键点描述子的匹配，有匹配的加入到当前帧特征点对应的地图点中
 *
 * 在局部地图中查找在当前帧视野范围内的点，将视野范围内的点和当前帧的特征点进行投影匹配
 */
void Tracking::SearchLocalPoints()
{
// 步骤1：遍历当前帧的mvpMapPoints，标记这些MapPoints不参与之后的搜索
    // 因为当前的mvpMapPoints一定在当前帧的视野中
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;// 当前帧的地图点
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();// 更新能观测到该点的帧数加1
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;// 标记该点被当前帧观测到
                pMP->mbTrackInView = false;// 标记该点将来不被投影，因为已经匹配过
            }
        }
    }

    int nToMatch=0;

// 步骤2：将所有局部MapPoints投影到当前帧，判断是否在视野范围内，然后进行投影匹配
    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        // 已经被当前帧观测到MapPoint不再判断是否能被当前帧观测到
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // 步骤2.1：判断LocalMapPoints中的点是否在在视野内
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5)) //cos(60) = 0.5
        {
            pMP->IncreaseVisible(); // 观测到该点的帧数加1，该MapPoint在某些帧的视野范围内
            nToMatch++;             // 只有在视野范围内的MapPoints才参与之后的投影匹配
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);// 0.8  最短的距离和次短的距离比值差异
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // 如果不久前进行过重定位，那么进行一个更加宽泛的搜索，阈值需要增大
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        // 步骤2.2：对视野范围内的MapPoints通过投影进行特征点匹配
        // 在局部地图点中搜寻和当前帧特征点描述子匹配的地图点,加入到当前帧特征点对应的地图点中
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

// 更新局部地图(与当前帧相关的帧和地图点) 用于局部地图点的跟踪 : 关键帧 + 地图点
// 更新局部关键帧-------局部地图的一部分，共视化程度高的关键帧、子关键帧、父关键帧
// 局部地图点的更新比较容易，完全根据局部关键帧来，所有局部关键帧的地图点就构成局部地图点
/**
 * @brief 断当前帧是否为关键帧
 * @return true if needed
 */
void Tracking::UpdateLocalMap()
{
    // This is for visualization 可视化
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();//更新关键帧
    UpdateLocalPoints();//更新地图点
}

/**
 * @brief 更新局部关键点，called by UpdateLocalMap()
 *  更新局部地图点
 * 局部地图点的更新比较容易，完全根据局部关键帧来，所有局部关键帧的地图点就构成局部地图点
 * 局部关键帧mvpLocalKeyFrames的MapPoints，更新mvpLocalMapPoints
 */
void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)//每一个局部关键帧的地图点
        {
            MapPoint* pMP = *itMP;// 每一个 局部关键帧
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)// 已经更新过了
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

// 更新局部关键帧-------局部地图的一部分
// 如何去选择当前帧对应的局部关键帧
    // 始终限制关键数量不超过80
    // 可以修改,这里比较耗时
    // 但是改小,精度可能下降
/*
 当关键帧数量较少时(<=80)，考虑加入第二部分关键帧，
 是与第一部分关键帧联系紧密的关键帧，并且始终限制关键数量不超过80。
 联系紧密体现在三类：
 1. 共视化程度高的关键帧 : 观测到当前帧地图点次数多的关键帧；
 2. 子关键帧；
 3. 父关键帧。

还有一个关键的问题是：如何判断该帧是否关键帧，以及如何将该帧转换成关键帧？
调用
NeedNewKeyFrame() 和
CreateNewKeyFrame()  两个函数来完成。
 */
/**
 * @brief 更新局部关键帧，called by UpdateLocalMap()
 *
 * 遍历当前帧的MapPoints，将观测到这些MapPoints的关键帧和相邻的关键帧取出，更新mvpLocalKeyFrames
 */
void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    // 更新地图点的观测帧
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)// 遍历当前帧的地图点
    {
        if(mCurrentFrame.mvpMapPoints[i])//当前帧的地图点不为NULL
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())// 被观测到
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;// 地图点的观测帧 观测地图点次数++
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;//未观测到地图点清除
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();// 局部关键帧清空
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
//  1. 共视化程度高的关键帧，观测到当前帧地图点次数多的关键帧；
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;//地图点的关键帧

        if(pKF->isBad())
            continue;

        if(it->second>max)// 观测到地图点数量最多的关键帧
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);// 保存局部关键帧
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        // 始终限制关键数量不超过80
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;
        // 根据权重w  二分查找有序序列中的某些对象
        // 返回前 w个 有序关键帧
        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);// 加入局部关键帧
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }
// 2. 子关键帧；
        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);// 加入局部关键帧
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }
// 3. 父关键帧
        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);// 加入局部关键帧
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

// 重定位
/*
 重定位Relocalization的过程大概是这样的：
1. 计算当前帧的BoW映射；
2. 在关键帧数据库中找到相似的候选关键帧；
3. 通过BoW匹配当前帧和每一个候选关键帧，如果匹配数足够 >15，进行EPnP求解；
4. 对求解结果使用BA优化，如果内点较少，则反投影候选关键帧的地图点到当前帧,获取额外的匹配点
     根据特征所属格子和金字塔层级重新建立候选匹配，选取最优匹配；
     若这样依然不够，放弃该候选关键帧，若足够，则将通过反投影获取的额外地图点加入，再进行优化。
5. 如果内点满足要求(>50)则成功重定位，将最新重定位的id更新：mnLastRelocFrameId = mCurrentFrame.mnId;　　否则返回false。
 */
/**
 * @brief 更新LocalMap
 *
 * 局部地图包括： \n
 * - K1个关键帧、K2个临近关键帧和参考关键帧
 * - 由这些关键帧观测到的MapPoints
 */
bool Tracking::Relocalization()
{
// 1. 计算当前帧的BoW映射； Compute Bag of Words Vector
    // 词典 N个M维的单词
    // 一帧的描述子  n个M维的描述子
    // 生成一个 N*1的向量 记录一帧的描述子 使用词典单词的情况
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
// 2. 在关键帧数据库中找到相似的候选关键帧；
    // 计算帧描述子 词典单词线性表示的词典单词向量，
    // 并和关键帧数据库中，每个关键帧的线性表示向量求距离，距离最近的一些帧为候选关键帧
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);
    if(vpCandidateKFs.empty())
        return false;
    const int nKFs = vpCandidateKFs.size();// 总的候选关键帧

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);// 描述子匹配器，最小距离 < 0.75*次短距离
    vector<PnPsolver*> vpPnPsolvers;//两关键帧之间的匹配点 Rt 求解器
    vpPnPsolvers.resize(nKFs);// 当前帧和每个候选关键帧都有一个求解器
    // 当前帧的关键点描述子 和 每个候选关键帧地图点 描述子的匹配点
    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;// 候选关键帧与当前帧匹配好坏标志
    vbDiscarded.resize(nKFs);

    int nCandidates=0;
    for(int i=0; i<nKFs; i++)// 关键帧数据库中每一个候选关键帧
    {
        KeyFrame* pKF = vpCandidateKFs[i];// 每一个候选关键帧
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
// 3. 通过BoW匹配当前帧和每一个候选关键帧，如果匹配数足够 >15，进行EPnP求解；
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else// 匹配数足够 >15   加入求解器
            {
                // 生成求解器
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                // 随机采样
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;// 添加求解器
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    // 直到找到一个候选匹配关键帧和符合变换关系Rt的足够的内点数量
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])// 跳过匹配效果差的 候选关键帧
                continue;

            // Perform 5 Ransac Iterations 5次随机采样序列,求解位姿Tcw
            vector<bool> vbInliers;// 符合变换的内点个数
            int nInliers;
            bool bNoMore;
            //求解器求解 进行EPnP求解
            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);//迭代5次 得到变换矩阵

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)//迭代5次效果还不好
            {
                vbDiscarded[i]=true;// EPnP求解不好，匹配效果差，放弃该候选关键帧
                nCandidates--;
            }
// 4. 对求解结果使用BA优化，如果内点较少，则反投影候选关键帧的地图点到当前帧获取额外的匹配点；
// 若这样依然不够，放弃该候选关键帧，若足够，则将通过反投影获取的额外地图点加入，再进行优化。
            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;// 地图点
                const int np = vbInliers.size();
                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])// 每一个符合的内点
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];// 对应的地图点 i帧  j帧下的地图点
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                // 使用BA优化位姿，返回优化较好效果较好的3d-2d优化边
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)// 这一候选帧匹配优化后效果不好
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])// 优化后更新状态 外点
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);// 地图点为空指针

                // If few inliers, search by projection in a coarse window and optimize again
                // 如果内点较少，则反投影候选关键帧的地图点vpCandidateKFs[i] 到 当前帧像素坐标系下
                // 根据格子和金字塔层级信息 在当前帧下选择与地图点匹配的特征点
                // 获取额外的匹配点
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        // 当前帧特征点对应的地图点数大于50进行优化
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);// 返回内点数量

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            // 缩小搜索窗口，再反投影
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                /// 再进行BA优化
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }

                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;// 重定位 帧ID
        return true;
    }

}

void Tracking::Reset()
{

    cout << "系统重置 System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "重置局部建图 Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "重置回环检测 Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "重置数据库 Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}



} //namespace ORB_SLAM
