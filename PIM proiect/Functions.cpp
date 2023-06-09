#include "Header.h"


unsigned char* processImage(unsigned char* img, int w, int h)
{
	unsigned char* result = new unsigned char[w * h];
	Mat inMat(h, w, CV_8UC1, img);
	Mat intermediaryMat;
	Mat processedMat(h, w, CV_8UC1, result);
	Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	GaussianBlur(inMat, intermediaryMat, cv::Size(0, 0), 3, 0);
	filter2D(intermediaryMat, processedMat, CV_8UC1, kernel);
	return result;
}

unsigned char* thresholdImage(unsigned char* img, int w, int h)
{
	unsigned char* result = new unsigned char[w * h];
	Mat inMat(h, w, CV_8UC1, img);
	Mat tresholdMat(h, w, CV_8UC1, result);
	threshold(inMat, tresholdMat, 130, 255, THRESH_BINARY);
	return result;
}

void watershed_v1(unsigned char* img, int w, int h) {
	Mat dist;
	Mat resultMat(h, w, CV_8UC1);

	unsigned char* intermediary = processImage(img, w, h);
	unsigned char* trsh = thresholdImage(intermediary, w, h);
	Mat processed(h, w, CV_8UC1, trsh);

	distanceTransform(processed, dist, DIST_L2, 3);
	normalize(dist, dist, 0.1, 12, NORM_MINMAX);
	//imshow("Distance Transform Image", dist);

	threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);


	Mat kernel1 = Mat::ones(3, 3, CV_8U);
	dilate(dist, dist, kernel1);
	//imshow("Peaks", dist);

	Mat dist_8u;
	dist.convertTo(dist_8u, CV_8U);

	vector<vector<Point> > contours;
	findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	Mat markers = Mat::zeros(dist.size(), CV_32S);
	for (size_t i = 0; i < contours.size(); i++)
	{
		drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i) + 1), -1);
	}
	circle(markers, Point(5, 5), 3, Scalar(255), -1);
	Mat markers8u;
	markers.convertTo(markers8u, CV_8U, 10);
	//imshow("Markers", markers8u);

	cvtColor(processed, processed, COLOR_GRAY2BGR);
	watershed(processed, markers);
	Mat mark;
	markers.convertTo(mark, CV_8U);
	bitwise_not(mark, mark);
	//imshow("Markers_v2", mark);

	vector<Vec3b> colors;
	for (size_t i = 0; i < contours.size(); i++)
	{
		int b = theRNG().uniform(0, 256);
		int g = theRNG().uniform(0, 256);
		int r = theRNG().uniform(0, 256);
		colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
	}
	resultMat = Mat::zeros(markers.size(), CV_8UC3);
	for (int i = 0; i < markers.rows; i++)
	{
		for (int j = 0; j < markers.cols; j++)
		{
			int index = markers.at<int>(i, j);
			if (index > 0 && index <= static_cast<int>(contours.size()))
			{
				resultMat.at<Vec3b>(i, j) = colors[index - 1];
			}
		}
	}

	imshow("Clasificare Tumori", resultMat);
}

void watershed_v2(unsigned char* img, int w, int h) {

	cv::Mat in(h, w, CV_8UC1, img);

	//1: segmentare grosiera a imaginii
	cv::Mat thrsh(h, w, CV_8UC1);
	cv::threshold(in, thrsh, 161, 255, cv::THRESH_OTSU);
	//cv::bitwise_not(thrsh, thrsh);
	//cv::imshow("1", thrsh);

	//2: corectarea erorilor de segmentare - inchidere, deschidere
	cv::Mat morph(h, w, CV_8UC1);
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	cv::morphologyEx(thrsh, morph, cv::MORPH_CLOSE, kernel);
	cv::morphologyEx(morph, morph, cv::MORPH_OPEN, kernel);
	//cv::imshow("2", morph);
	
	//3: identificarea regiunilor care apartin 'background-ului' - dilatare morfologica
	cv::Mat dilated(h, w, CV_8UC1);
	cv::morphologyEx(morph, dilated, cv::MORPH_DILATE, kernel);
	//cv::imshow("3", dilated);

	//4: determinarea transformatei distanta
	cv::Mat distance(h, w, CV_8UC1);
	cv::distanceTransform(morph, distance, cv::DIST_L2, 5);
	cv::normalize(distance, distance, 1, 0, cv::NORM_INF);
	//cv::imshow("4", distance);

	//5: identificarea regiunilor care apartin interioarelor obiectelor cautate
	cv::Mat interiors(h, w, CV_8UC1);
	double min, max;
	cv::minMaxLoc(distance, &min, &max, NULL, NULL);
	cv::threshold(distance, interiors, 0.05 * max, 255, cv::THRESH_BINARY);
	//cv::imshow("5", interiors);
	
	//6: separam regiunile anterioare pe obiecte
	Mat interiors_8u, norm_components, components;
	interiors.convertTo(interiors_8u, CV_8U);
	int labels = connectedComponents(interiors_8u, components, 8);
	normalize(components, norm_components, 0, 255, NORM_MINMAX, CV_8U);
	norm_components.convertTo(norm_components, CV_8UC1);
	//cv::imshow("6", norm_components);

	//7: obtinere regiuni de incertitudine
	dilated.setTo(0, norm_components);
	//cv::imshow("7", dilated);

	//8: matricea completa a markerelor
	Mat final_markers;
	final_markers = norm_components + 1;
	final_markers.setTo(0, dilated);
	//cv::imshow("8", final_markers);

	//9: utilizam functia watershed din OpenCV
	cv::Mat in_rgb;
	cv::cvtColor(in, in_rgb, cv::COLOR_GRAY2RGB);
	final_markers.convertTo(final_markers, CV_32SC1);
	cv::watershed(in_rgb, final_markers);
	in_rgb.setTo(0, final_markers == -1);
	cv::imshow("Identificare Tumori", in_rgb);

	cout << labels; 
	cv::Mat final_markers_8u;
	final_markers.convertTo(final_markers_8u, CV_8U);
	cv::bitwise_not(final_markers_8u, final_markers_8u);
	vector<Vec3b> colors;
	for (size_t i = 0; i <= labels; i++)
	{
		int b = theRNG().uniform(0, 256);
		int g = theRNG().uniform(0, 256);
		int r = theRNG().uniform(0, 256);
		colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
	}

	cv::Mat result = Mat::zeros(final_markers.size(), CV_8UC3);
	for (int i = 0; i < final_markers.rows; i++)
	{
		for (int j = 0; j < final_markers.cols; j++)
		{
			int index = final_markers.at<int>(i, j);
			if (index > 0 && index <= static_cast<int>(labels))
			{
				result.at<Vec3b>(i, j) = colors[index - 1];
			}
		}
	}

	imshow("Rezultat Final", result);

}


