#include <string>
#include <fstream>
#include <streambuf>
#include <algorithm>

#include <boost/thread.hpp>
#include <boost/chrono.hpp>

#include <set>
#include <tuple>
#include <time.h>
#include <queue>

#include <ros/ros.h>

#include <openpose/OpenPose.hpp>

#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <vision_msgs/OpenPoseRecognitions.h>
#include <vision_msgs/OpenPoseRecognize.h>

//Node config
DEFINE_bool(debug_mode, true, "The debug mode");
DEFINE_int32(frame_rate, 30, "The frame rate");
DEFINE_bool(publish_image, true, "Publish result image");
DEFINE_int32(logging_level, 3, "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for low priority messages and 4 for important ones.");
DEFINE_string(rgb_camera_topic, "/usb_cam/image_raw", "The rgb input camera topic.");
DEFINE_string(openpose_service, "openpose/recognize", "The openpose service recognizer.");
DEFINE_string(result_pose_topic, "openpose/result", "The result pose topic.");
// OpenPose
DEFINE_string(model_folder, "/opt/openpose/models/",      "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(model_pose, "COCO", "Model to be used (e.g. COCO, MPI, MPI_4_layers).");
DEFINE_string(net_resolution, "640x480", "Multiples of 16. If it is increased, the accuracy potentially increases. If it is decreased, the speed increases. For maximum speed-accuracy balance, it should keep the closest aspect ratio possible to the images or videos to be processed. E.g. the default `656x368` is optimal for 16:9 videos, e.g. full HD (1980x1080) and HD (1280x720) videos.");
DEFINE_string(resolution, "640x480", "The image resolution (display and output). Use \"-1x-1\" to force the program to use the default images resolution.");
DEFINE_int32(num_gpu_start, 0, "GPU device start number.");
DEFINE_double(scale_gap, 0.3, "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1. If you want to change the initial scale, you actually want to multiply the `net_resolution` by your desired initial scale.");
DEFINE_int32(scale_number, 1, "Number of scales to average.");
// OpenPose Rendering
DEFINE_bool(disable_blending, false, "If blending is enabled, it will merge the results with the original frame. If disabled, it will only display the results.");
DEFINE_double(render_threshold, 0.05, "Only estimated keypoints whose score confidences are higher than this threshold will be rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;  while small thresholds (~0.1) will also output guessed and occluded keypoints, but also  more false positives (i.e. wrong detections).");
DEFINE_double(alpha_pose, 0.6, "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will hide it. Only valid for GPU rendering.");

OpenPose * openPoseEstimator_ptr;
ros::NodeHandle * nh_ptr;
ros::Publisher pubResult_;
ros::Publisher pubImage_;
bool viewImage_ = false;
double seconds;
double fps;
time_t start, end;
long numFrames = 0;

//std::queue<std::function<void()>> queue;
std::queue<std::function<std::vector<vision_msgs::OpenPoseRecognition>()>> queue;
std::queue<vision_msgs::OpenPoseRecognitions> serviceQueue;
bool serviceFinish = false;
bool serviceProcessing = false;
boost::mutex mutex;
void recognitionThread()
{
	mutex.lock();
	std::string modelFoler = (std::string) FLAGS_model_folder;
	op::PoseModel poseModel =  op::flagsToPoseModel(FLAGS_model_pose);
	op::Point<int> netResolution = op::flagsToPoint(FLAGS_net_resolution);
	op::Point<int> outputSize = op::flagsToPoint(FLAGS_resolution);
	int numGpuStart = (int) FLAGS_num_gpu_start;
	float scaleGap = (float) FLAGS_scale_gap;
	float scaleNumber = (float) FLAGS_scale_number;
	bool disableBlending = (bool) FLAGS_disable_blending;
	float renderThreshold = (float) FLAGS_render_threshold;
	float alphaPose = (float) FLAGS_alpha_pose;
	
	openPoseEstimator_ptr = new OpenPose();
	openPoseEstimator_ptr->initOpenPose(modelFoler, poseModel, netResolution, outputSize, numGpuStart, scaleGap, scaleNumber, disableBlending, renderThreshold, alphaPose);
	cv::Mat initFrame = cv::Mat::zeros(640, 480, CV_8UC3);
	cv::Mat opResult;
	std::vector<std::map<int, std::vector<float> > > keyPoints;
	openPoseEstimator_ptr->framePoseEstimation(initFrame, opResult, keyPoints);
	mutex.unlock();
	time(&start);
	std::cout << "open_pose_node.->The openpose node has bee initialized" << std::endl;
	while(ros::ok())
	{
		if(!queue.empty())
		{
			//mutex.lock();
			std::function<std::vector<vision_msgs::OpenPoseRecognition>()> func = queue.front();
                	queue.pop();
			//mutex.unlock();
                	func();
		}
		//boost::this_thread::sleep(boost::posix_time::milliseconds(1));
	}
	delete openPoseEstimator_ptr;
}

std::vector<vision_msgs::OpenPoseRecognition> getRecognitions(sensor_msgs::Image image, bool publishImage, bool serviceProcessing)
{
	cv_bridge::CvImagePtr cam_image;
	std::vector<vision_msgs::OpenPoseRecognition> recognitions;
	std_msgs::Header imageHeader_;
	try {
		cam_image = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
		imageHeader_ = image.header;
	} catch (cv_bridge::Exception& e) {
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return recognitions;
	}

	cv::Mat opResult;
	std::vector<std::map<int, std::vector<float> > > keyPoints;
	openPoseEstimator_ptr->framePoseEstimation(cam_image->image, opResult, keyPoints);
	for(int i = 0; i < keyPoints.size(); i++)
	{
		vision_msgs::OpenPoseRecognition recognition;
		std::map<int, std::vector<float> > poses = keyPoints[i];
		for(std::map<int, std::vector<float> >::iterator it = poses.begin(); it != poses.end(); it++)
		{
			vision_msgs::OpenPoseKeyPoint keyPoint;
			keyPoint.x = it->second[0];
			keyPoint.y = it->second[1];
			keyPoint.score = it->second[2];
			recognition.keyPoints.push_back(keyPoint);
		}
		recognitions.push_back(recognition);
	}
	vision_msgs::OpenPoseRecognitions result;
	result.recognitions = recognitions;
	if(serviceProcessing)
		serviceQueue.push(result);
	else
		pubResult_.publish(result);

	if(FLAGS_publish_image)
	{
        	cv_bridge::CvImage cvImage;
        	cvImage.encoding = sensor_msgs::image_encodings::BGR8;
        	cvImage.image = opResult;
        	pubImage_.publish(*cvImage.toImageMsg());
	}

		
	
	if (FLAGS_debug_mode)
	{
		cv::waitKey(1);
		cv::imshow("Openpose estimation", opResult);
		time(&end);
		numFrames++;
		seconds = difftime(end, start); 
		fps = numFrames / seconds;
		std::cout << "Estimated frames per second:" << fps << std::endl;
	}
	return recognitions;
}

void cameraCallback(sensor_msgs::Image msg)
{
	std::function<std::vector<vision_msgs::OpenPoseRecognition>()> func = std::bind(getRecognitions, msg, true, false);
	mutex.lock();
	queue.push(func);
	mutex.unlock();
}

bool recognizeCallback(vision_msgs::OpenPoseRecognize::Request &req, vision_msgs::OpenPoseRecognize::Response &res)
{
	std::function<std::vector<vision_msgs::OpenPoseRecognition>()> func = std::bind(getRecognitions, req.input_image, true, true);
	//mutex.lock();
	queue.push(func);
	//mutex.unlock();
	
	while(ros::ok() && serviceQueue.empty());
	
	vision_msgs::OpenPoseRecognitions resul = serviceQueue.front();
	serviceQueue.pop();
	res.recognitions = resul.recognitions;
	
	return true;
}

void rosThread()
{
	ros::Rate rate((int) FLAGS_frame_rate);
	ros::Subscriber imageSubscriber = nh_ptr->subscribe((std::string) FLAGS_rgb_camera_topic, 1, cameraCallback);
	ros::ServiceServer recognizeService = nh_ptr->advertiseService((std::string) FLAGS_openpose_service, recognizeCallback);
	pubResult_ = nh_ptr->advertise<vision_msgs::OpenPoseRecognitions>((std::string) FLAGS_result_pose_topic, 1);
	pubImage_ = nh_ptr->advertise<sensor_msgs::Image>("openpose/result_image", 1);
	while(ros::ok())
	{
		rate.sleep();
		ros::spinOnce();
	}
}

int main(int argc, char ** argv){

	ros::init(argc, argv, "openpose_node");
	std::cout << "open_pose_node.->Initializing the openpose node by Rey" << std::endl;
	ros::NodeHandle nh;
	nh_ptr = &nh;

	if(ros::param::has("~debug_mode"))
		ros::param::get("~debug_mode", FLAGS_debug_mode);
	if(ros::param::has("~publish_image"))
		ros::param::get("~publish_image", FLAGS_publish_image);
	if(ros::param::has("~frame_rate"))
		ros::param::get("~frame_rate", FLAGS_frame_rate);
	if(ros::param::has("~model_folder"))
		ros::param::get("~model_folder", FLAGS_model_folder);
	if(ros::param::has("~model_pose"))
		ros::param::get("~model_pose", FLAGS_model_pose);
	if(ros::param::has("~net_resolution"))
		ros::param::get("~net_resolution", FLAGS_net_resolution);
	if(ros::param::has("~resolution"))
		ros::param::get("~resolution", FLAGS_resolution);
	if(ros::param::has("~num_gpu_start"))
		ros::param::get("~num_gpu_start", FLAGS_num_gpu_start);
	if(ros::param::has("~scale_gap"))
		ros::param::get("~scale_gap", FLAGS_scale_gap);
	if(ros::param::has("~scale_number"))
		ros::param::get("~scale_number", FLAGS_scale_number);
	if(ros::param::has("~render_threshold"))
		ros::param::get("~render_threshold", FLAGS_render_threshold);
	if(ros::param::has("~rgb_camera_topic"))
		ros::param::get("~rgb_camera_topic", FLAGS_rgb_camera_topic);
	if(ros::param::has("~openpose_service"))
		ros::param::get("~openpose_service", FLAGS_openpose_service);
	if(ros::param::has("~result_pose_topic"))
		ros::param::get("~result_pose_topic", FLAGS_result_pose_topic);

	std::cout << "open_pose_node.->The node will be initializing with the next parameters" << std::endl;
	std::cout << "open_pose_node.->Debug mode:" << FLAGS_debug_mode << std::endl;
	std::cout << "open_pose_node.->Publish image:" << FLAGS_publish_image << std::endl;
	std::cout << "open_pose_node.->Model folder:" << FLAGS_model_folder << std::endl;
	std::cout << "open_pose_node.->Model pose:" << FLAGS_model_pose << std::endl;
	std::cout << "open_pose_node.->Net resolution:" << FLAGS_net_resolution << std::endl;
	std::cout << "open_pose_node.->Resolution:" << FLAGS_resolution << std::endl;
	std::cout << "open_pose_node.->Num gpu start:" << FLAGS_num_gpu_start << std::endl;
	std::cout << "open_pose_node.->Scale gap:" << FLAGS_scale_gap << std::endl;
	std::cout << "open_pose_node.->Scale number:" << FLAGS_scale_number << std::endl;
	std::cout << "open_pose_node.->Render threshold:" << FLAGS_render_threshold << std::endl;
	std::cout << "open_pose_node.->rgb camera topic:" << FLAGS_rgb_camera_topic << std::endl;
	std::cout << "open_pose_node.->openpose service:" << FLAGS_openpose_service << std::endl;
	std::cout << "open_pose_node.->Result pose topic:" << FLAGS_result_pose_topic << std::endl;

	op::log("OpenPose ROS Node", op::Priority::High);
	std::cout << "OpenPose->loggin_level_flag:" << FLAGS_logging_level << std::endl; 
	op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.", __LINE__, __FUNCTION__, __FILE__);
	
	boost::thread thread1{recognitionThread};
	boost::thread thread2{rosThread};
  	thread1.join();
  	thread2.join();

	return 1;
}
