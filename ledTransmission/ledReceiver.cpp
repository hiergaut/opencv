

#include <opencv2/opencv.hpp>
#include <unistd.h>

#define FFPS 10

#define PERIOD 1000 /(float)FFPS // millisecond
#define SIZE width *height

int width;
int height;

using namespace std;

void findBlinkLed(cv::VideoCapture cap, int* pos) {
	cv::Mat m, m2;
	cap >> m;
	usleep(PERIOD);
	cap >> m2;

	cv::cvtColor(m, m, cv::COLOR_BGR2GRAY);
	cv::cvtColor(m2, m2, cv::COLOR_BGR2GRAY);

	cv::imshow("w", m);
	cv::moveWindow("w", 20, 20);

	uchar* p =m2.data;
	for (int i =0; i <SIZE; i++) {
		*p++ =127;
	}

	cv::imshow("w2", m2);
	cv::moveWindow("w2", 20, 520);

	while (cv::waitKey(10) != 27);
}


int main(int argc, char **argv) {
	cv::VideoCapture cap(0);
	if (! cap.isOpened())
		return -1;
	
	// cap.set(CV_CAP_PROP_FPS, 90);
	// cap.set(3, 320);
	// cap.set(4, 240);
	width =cap.get(cv::CAP_PROP_FRAME_WIDTH);
	height =cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	cout << "resolution : " << width << "x" << height << endl;
	cout << "period : " << PERIOD << endl;

	int pos =-1;
	findBlinkLed(cap, &pos);




	cap.release();
    return 0;
}