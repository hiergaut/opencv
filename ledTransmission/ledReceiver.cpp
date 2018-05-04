
#include <assert.h>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <unistd.h>
// #include <set>
// #include <list>

#define FFPS 40
#define THRESH 235

#define PERIOD 1000000000 / FFPS // microseconds
#define SIZE width *height * 3

// #define DEBUG_MODE

int width;
int height;

using namespace std;

// typedef chrono::high_resolution_clock Clock;

void findBlinkLed(cv::VideoCapture cap, int *pos) {

	bool found = false;
	cv::Mat m, m2;
	// int cpt =0;
	// auto t0 = chrono::high_resolution_clock::now();
	// auto step =chrono::microseconds(PERIOD);
	// set<int> best;
	int occur[SIZE];
	for (int i = 0; i < SIZE; i++) {
		occur[i] = 0;
		// cout << occur[i];
	}
	// printf("%o", best.end());
	while (not found) {
		// for (int i =0; i <10; i++) {
		// 	cap >> m;
		// }
		// auto start = std::chrono::high_resolution_clock::now();
		auto until =
			chrono::high_resolution_clock::now() + chrono::nanoseconds(PERIOD);

		// cout << "start" << endl;
		// cout <<
		// "start in " << static_cast<chrono::duration<double>>(end).count()
		cap >> m;
		// << endl;
		// assert(! m.empty());
		// while (chrono::system_clock::now() < end);
		this_thread::sleep_until(until);

		cap >> m2;
		// cout << "end" << endl;
		// sleep(1);
		// usleep(PERIOD *1.5);
		// assert(! m2.empty());

		// cv::cvtColor(m, m, cv::COLOR_BGR2GRAY);
		// cv::cvtColor(m2, m2, cv::COLOR_BGR2GRAY);

		uchar *p = m.data;
		uchar *p2 = m2.data;

		// int posGap = -1;
		int maxGap = -1;
		int posGap = -1;
		for (int i = 0, gap; i < SIZE; i++) {
			// *p++ =127;
			// *p3++ =(abs(*p++ -*p2++) > 127) *255;
			gap = abs(*p++ - *p2++);
			if (gap > maxGap) {
				maxGap = gap;
				posGap = i;

				// if (maxGap > 250)
				// found =*pos == i;
				// *pos = i;
			}
		}

		if (maxGap > THRESH) {
			if (occur[posGap]++ > 10) {
				*pos = posGap;
				found = true;
			}
			// if (best.find(posGap) != best.end()) {
			// 	best.insert(posGap);
			// } else {
			// 	*pos = posGap;
			// 	found = true;
			// }
		}
		// found = maxGap > 220;
		cout << "\rmaxGap = " << maxGap << " " << flush;

		// m =m2.clone();

		// cv::imshow("w2", m2);
		// cv::moveWindow("w2", 20, 520);

		// if (maxGap > 250) {
		// 	int x = (*pos / 3) % width;
		// 	int y = (*pos / 3) / width;
		// 	cv::putText(m, "+", cv::Point2f(x - 12, y + 15),
		// 				cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 1);
		// }

		// cv::imshow("w", m);
		// cv::moveWindow("w", 20, 20);

		// if (cv::waitKey(1) == 27)
		// 	exit(0);
	}

	// #ifdef DEBUG
	int x = (*pos / 3) % width;
	int y = (*pos / 3) / width;
	// x =100;
	// y =100;
	// cout << m;
	cv::putText(m, "+", cv::Point2f(x - 15, y + 10), cv::FONT_HERSHEY_SIMPLEX,
				1, cv::Scalar(0, 255, 0), 1);
	cv::imshow("w", m);
	cv::moveWindow("w", 20, 20);
	while (cv::waitKey(100) != 27)
		;
	// #endif
	// exit(0);
}

void usage() {
	cout << "usage: ledReceiver [options] device\n"
		 << "options :\n"
		 << "	-v <num>   video device number /dev/video*\n"
		 << "	-i         init sensor, find led position\n"
		 << endl;
}

int main(int argc, char **argv) {
	int device = 0;
	int init = 0;

	int option;
	while ((option = getopt(argc, argv, "d:hi")) != -1) {
		switch (option) {
		case 'i':
			init = 1;
			break;

		case 'h':
			usage();
			exit(EXIT_SUCCESS);

		case 'd':
			device = atoi(optarg);
			break;

		default:
			usage();
			exit(EXIT_FAILURE);
		}
	}

	cv::VideoCapture cap(device);
	if (!cap.isOpened())
		return -1;

	// cap.set(CV_CAP_PROP_FPS, FFPS);
	cap.set(CV_CAP_PROP_FPS, 100);

	// cap.set(3, 1280);
	// cap.set(4, 720);
	width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
#ifdef DEBUG
	cout << "resolution : " << width << "x" << height << endl;
#endif
	// cout << "period : " << PERIOD << " us" << endl;

	int pos = -1;
	if (init) {
		findBlinkLed(cap, &pos);
		ofstream file;
		file.open("ledPos.txt");
		cout << "\nled position = " << pos << endl;
		file << pos;
		file.close();
		exit(EXIT_SUCCESS);
	}

	ifstream file("ledPos.txt");
	if (!file.good()) {
		cout << "init led sensor before launch transmit" << endl;
		exit(EXIT_FAILURE);
	}

	file >> pos;

	assert(pos != -1);
#ifdef DEBUG
	cout << "led position: " << pos << endl;
#endif

	// findBlinkLed(cap, &pos);
	cv::Mat m;
	// uchar *led = &m.data[0] +pos;

	// uchar* prev =m.data;
	int c;
	int bit;
	int step = 10;
	auto t0 = chrono::high_resolution_clock::now();
	auto t = chrono::nanoseconds(PERIOD);
	int cpt = 1;
	bool waitBitStart = true;
	unsigned char str[256];
	char istr = 0;
	while (true) {
#ifdef DEBUG
		int64 start = cv::getTickCount();
#endif
		for (int j = 0; j < step; j++) {
			// start:
			c = 0;
			for (int i = 0; i < 8; i++) {
				// assert(cap.grab());
				cap >> m;
				// assert(!m.empty());
				// cv::cvtColor(m, m, CV_BGR2GRAY);
				bit = m.data[pos] > THRESH;
				c = c << 1;
				c = c | bit;

				if (waitBitStart && bit == 1) {
					c = 0;
					i = -1;
					bit = 0;
					waitBitStart = false;
#ifdef DEBUG
					cout << "\n\033[1;32m*************** START BIT\033[0m"
						 << endl;
					start = cv::getTickCount();
#endif
					j = 0;
					istr = 0;
					// goto start;
				}
#ifdef DEBUG
				else {
					// bit =*led > 127;
					cout << bit;
					printf(" %3d ",int(m.data[pos]));
				}
#endif

				// assert(m.data == prev);

				// cout <
				// prev =m.data;
				// cout << *led << endl;
				// auto until =t0 +step;
				this_thread::sleep_until(t0 + cpt++ * t);
			}
#ifdef DEBUG
			cout << " " << c << endl;
#endif
			str[istr++] = c;
			if (!waitBitStart && c == 0) {
#ifdef DEBUG
				cout << "\033[1;33m*************** END BIT\033[0m  str: '"
					 << str << "'" << endl;
				waitBitStart = true;
#else
				cout << str << endl;
				exit(EXIT_SUCCESS);
#endif
			}
		}
#ifdef DEBUG
		float diff = (cv::getTickCount() - start) / cv::getTickFrequency();
		cout << "BPS = " << step * 8 / diff << endl;
#endif

		// cout << " " << c << " " << static_cast<unsigned char>(c) << endl;
	}

	// cout << "pos led : " << pos << endl;

	cap.release();
	return 0;
}
