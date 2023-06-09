

#include "Header.h"


int main()
{
	Mat img = imread("C:/Users/Radu/source/repos/PIM-Proiect/PIM-Proiect/Resources/brain-tumor-2.jpeg", IMREAD_GRAYSCALE);
	if (img.empty())
	{
		cout << "Could not open or find the image" << endl;
		cin.get();
		return -1;
	}

	int w = img.cols;
	int h = img.rows;
	unsigned char* imgData = img.data;
	
	watershed_v1(imgData, w, h);

	watershed_v2(imgData, w, h);

	waitKey(0);
	return 0;
}