/*
Bassam Arshad
0259149

Project-04 SIFT Detector & Descriptors


*/


#include <opencv2/core/core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d/nonfree.hpp"
# include "opencv2/features2d/features2d.hpp"

using namespace cv;
using namespace std;

void   BFMatching(Mat img1, Mat img2, vector<cv::KeyPoint> keypoints1, vector<cv::KeyPoint> keypoints2, Mat descriptors_1, Mat descriptors_2,float ratio);
void   FLANNMatching(Mat img1, Mat img2, vector<cv::KeyPoint> keypoints1, vector<cv::KeyPoint> keypoints2, Mat descriptors_1, Mat descriptors_2,float ratio);

int main(int argc, const char* argv[])
{
	 Mat img1 = imread("stop1.jpg"); 
	 Mat img2 = imread("stop3.jpg");

	Mat gray1,gray2;

	//Convert to grayscale
	cvtColor(img1, gray1, COLOR_BGR2GRAY);
	cvtColor(img2, gray2, COLOR_BGR2GRAY);


	Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create(); 
	
	std::vector<cv::KeyPoint> keypoints1,keypoints2;
	Mat descriptors_1, descriptors_2;

	//Detect the KeyPoints
	sift->detect(gray1, keypoints1);
	sift->detect(gray2, keypoints2);

	//Compute the Feature Descriptors for the KeyPoints
	sift->compute(gray1, keypoints1, descriptors_1);
	sift->compute(gray2, keypoints2, descriptors_2);


	//Use Brute Force Matcher
	BFMatching(img1, img2, keypoints1, keypoints2, descriptors_1, descriptors_2,0.6);
	//Use FLANN Matcher
	FLANNMatching(img1, img2, keypoints1, keypoints2, descriptors_1, descriptors_2,0.6);


	// Add results to image and save.
	cv::Mat output; 
	cv::drawKeypoints(img1, keypoints1, output, Scalar::all(-1),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::imshow("sift_result.jpg", output);

	waitKey();

	return 0;
}

void   BFMatching(Mat img1, Mat img2, vector<cv::KeyPoint> keypoints1, vector<cv::KeyPoint> keypoints2, Mat descriptors_1, Mat descriptors_2,float r)
{
	//Using the below .match for the matcher gives a lot of results --> implemented the knn one , for better results.
	/*
	//Using Brute Force Matcher - To match the descriptors
	BFMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors_1, descriptors_2, matches);
	//matcher.radiusMatch(descriptors_1, descriptors_2, matches,2);
	*/
	
	std::vector<std::vector<cv::DMatch>> matches;
	cv::BFMatcher matcher;
	//k-nearest neighbor matcher
   matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);  // Find two nearest matches

	vector<cv::DMatch> good_matches;
	for (int i = 0; i < matches.size(); ++i)
	{
		const float ratio = r; // As in Lowe's paper; can be tuned
		if (matches[i][0].distance < ratio * matches[i][1].distance)
		{
			good_matches.push_back(matches[i][0]);
		}
	}
	
	Mat img_matches;
	//drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, CV_RGB(0, 255, 0), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Show detected matches
	namedWindow("BF-Matcher SIFT Matches", 0);
	imshow("BF-Matcher SIFT Matches", img_matches);
	
}


void   FLANNMatching(Mat img1, Mat img2, vector<cv::KeyPoint> keypoints1, vector<cv::KeyPoint> keypoints2, Mat descriptors_1, Mat descriptors_2,float r)
{

	
	FlannBasedMatcher matcher;
	std::vector<std::vector<cv::DMatch>> matches;
	//k-nearest neighbor matcher
	matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);  // Find two nearest matches

	vector<cv::DMatch> good_matches;
	for (int i = 0; i < matches.size(); ++i)
	{
		const float ratio = r; // As in Lowe's paper; can be tuned
		if (matches[i][0].distance < ratio * matches[i][1].distance)
		{
			good_matches.push_back(matches[i][0]);
		}
	}
  

	//-- Draw only "good" matches
	Mat img_matches;
	//drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, CV_RGB(0,255,0), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Show detected matches
	namedWindow("FLANN-Matcher SIFT Matches", 0);
	imshow("FLANN-Matcher SIFT Matches", img_matches);

}