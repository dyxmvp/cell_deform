// 07052015:
// 1. A bug is found: "vector<unsigned char> cineData(imageH * imageW);" should be defined as local variable, or will influence the results

// 07092015
// 1. minEllipse[0] = fitEllipse(contours[j]) should be revised to minEllipse[0] = fitEllipse(Mat(contours[j]));

// 07202015
// 1. Diameter function return equivalent diameter directly

// 07252015
// 1. Head files number is optimized
// 2. Kernel size is added as input 

// 11072015
// 1. Rule out the close cells
// 2. Improve the accuracy in deformability calculation

// 11192015
//1. Further improve the rule out the close cells
//2. Use class to simplied the code

#include "stdafx.h"
#include "PhCon.h"
#include "PhInt.h"
#include "PhFile.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <omp.h>  

using namespace std;
using namespace cv;


/** Global Variables */
///
int imageH = 0;
int imageW = 0;

// Parameters for Initial Diameter
int X0_dia = 0;
int Xlength_dia = 0;
int Y0_dia = 0;
int Ylength_dia = 0;
int Threshold1 = 0;
int Threshold2 = 0;
int Morph_Operator_dia = 0;
int Morph_Elem_dia = 0;
int Morph_Size_dia = 0;
int Kernel_Size_dia = 0;

// Parameters for Deformability
int Deform_Delay = 0;
int X0_def = 0;
int Xlength_def = 0;
int Y0_def = 0;
int Ylength_def = 0;
int Threshold_Deform = 0;
int Morph_Operator_def = 0;
int Morph_Elem_def = 0;
int Morph_Size_def = 0;

//
long double const PI = 3.14159265358979323846;

//
PSTR imageName = "AT1_450uLmin_M80S40_Step_r100_30_100000fps1.cine";// MMM "Z:\\Yanxiang\\2015\\11262015\\1\\231_450uLmin_M80S40_Step_r100_30_100000fps.cine"; //"CHO_450uLmin_M80S40_Step_r100_30_100000fps.cine";//"Z:\\Yanxiang\\2015\\11262015\\1\\231_450uLmin_M80S40_Step_r100_30_100000fps.cine"; //"t11294.cine";
int imageN = 0;
int image1 = 0;


//
class Cyto
{
public:
	// Set compute range

	void compute(int partNum, int ext);
	double GetMedian(double daArray[], int iSize);

private:
	int from;
	int to;

	double diameter;
	double deform1;
	double deform2;
	double deformations;
	double centroids;

	CINEHANDLE cineHandle;
	BITMAPINFOHEADER bitmapInfoHeader;
	UINT imgSizeInBytes;
	PBYTE m_pImageBuffer;

	ofstream sizefile, deformfile, centroidfile;
	string filename_dia, filename_def, filename_cen;

	double Initial_Diameter(int i);
	double Detect(int i);
	double Deformation(int i);
	int Detect_Def(int i);
	double Centroid(int i);

	Mat Morphology_Operations(Mat dst_binary, int morph_operator, int morph_elem, int morph_size);

};



///

/** Main */
int main(int argc, char** argv)
{
	
	CINEHANDLE cineHandle;
	UINT imgSizeInBytes;
	PBYTE m_pImageBuffer;

	int firstImage = 0;
	int lastImage = 0;
	
	// Read parameters from file
	std::fstream parameters("parameters.txt", std::ios_base::in);
	parameters >> imageH >> imageW >> X0_dia >>
		          Xlength_dia >> Y0_dia >> Ylength_dia >>
		          Threshold1 >> Threshold2 >>
		          Morph_Operator_dia >> Morph_Elem_dia >> Morph_Size_dia >>
				  Kernel_Size_dia >>
		          Deform_Delay >>
		          X0_def >> Xlength_def >> Y0_def >> Ylength_def >>
		          Threshold_Deform >>
		          Morph_Operator_def >> Morph_Elem_def >>
		          Morph_Size_def;

	// Initialization for PhSDK
	//PhRegisterClientEx(NULL, NULL, NULL, PHCONHEADERVERSION);
	PhStartPhFile();
	//PhLVRegisterClientEx(NULL, NULL, PHCONHEADERVERSION);


	// Read cine files
	PhNewCineFromFile(imageName, &cineHandle);
	PhGetCineSaveRange(cineHandle, &firstImage, &lastImage);
	PhGetCineInfo(cineHandle, GCI_MAXIMGSIZE, (PVOID)&imgSizeInBytes);
	m_pImageBuffer = (PBYTE)_aligned_malloc(imgSizeInBytes, 32);// 16);

	image1 = firstImage + 1;
	imageN = lastImage - firstImage + 1;
	
    
	
	Cyto ana1, ana2, ana3, ana4, ana5, ana6, ana7, ana8;
	
	//int pn = 0;
    
    #pragma omp parallel sections
	{
       #pragma omp section   // thread 1
	    ana1.compute(1, 0);
		
       #pragma omp section   // thread 2
		ana2.compute(2, 0);

       #pragma omp section   // thread 3
		ana3.compute(3, 0);

       #pragma omp section   // thread 4
		ana4.compute(4, 0);

       #pragma omp section   // thread 5
		ana5.compute(5, 0);

       #pragma omp section   // thread 5
		ana6.compute(6, 0);

       #pragma omp section   // thread 5
		ana7.compute(7, 0);

       #pragma omp section   // thread 5
		ana8.compute(8, -5);
	  
	}

	



	//waitKey(0);           // Wait for a keystroke in the window
	//cout << "Done!" << endl;
	
	return 0;
}


void Cyto::compute(int partNum, int ext)
{	
	//cout << partNum << ": " << omp_get_thread_num() << endl;
	
	// Read cine files
	PhNewCineFromFile(imageName, &cineHandle);
	PhGetCineInfo(cineHandle, GCI_MAXIMGSIZE, (PVOID)&imgSizeInBytes);
	m_pImageBuffer = (PBYTE)_aligned_malloc(imgSizeInBytes, 16);

	// Output files
	filename_dia = "diameters" + to_string(partNum) + ".txt";
	filename_def = "deformations" + to_string(partNum) + ".txt";
	filename_cen = "centroids" + to_string(partNum) + ".txt";
	sizefile.open(filename_dia);
	deformfile.open(filename_def);
	centroidfile.open(filename_cen);
	
	from = image1 + (partNum - 1) / 8.0 * imageN;   //change number according to the thread number
	to = image1 + partNum / 8.0 * imageN + ext;

	for (int i = from; i < to; ++i)
	{

		diameter = Initial_Diameter(i);
		//cout << i << "," << diameter << endl;

		// Test large size
		//if (diameter > 30)
		//{
		//	testfile_dia << i << ",  " << diameter << endl;
		//}

		if (diameter != 0)   // if diameter is good

		{
			if (Detect_Def(i + 2) == 0)
			{
				
				continue;
			}
			
			if (Detect(i - 1) != 0)
			{
				
				continue;
			}

			if (Detect(i + 1) != 0)
			{
				i = i + 1;
				continue;
			}

			if (Detect(i + 2) != 0)
			{
				i = i + 2;
				continue;
			}

			if (Detect(i + 3) != 0)
			{
				i = i + 3;
				continue;
			}

			deform1 = Deformation(i + Deform_Delay);

			//cout << deform1 << endl;
			
			// Test large size
			//	if (deform1 > 2 && deform1 < 3)
			//{
			//	testfile_def << i << ",  " << deform1 << endl;
			//	}

			deform2 = Deformation(i + Deform_Delay + 1);

			// Test large size
			//if (deform2 > 2 && deform2 < 3)
			//{
			//	testfile_def << i << ",  " << deform2 << endl;
			//}

			//cout << deform2 << endl;

			if (deform1 || deform2)
			{
				deformations = max(deform1, deform2);   // Store deformations 
				centroids = Centroid(i);
				//cout << "result: " << "i = " << i << ",   " << diameter; cout << ",   "; cout << deform1 << ",  " << deform2 << endl << endl;
				//cout << ",   "; cout << i;  cout << '\n'; // Display the results
				sizefile << diameter << endl; // "i = " << i << ",   " << diameter << ",   " << deform1 << ",   " << deform2 << endl; // "\t" << deformations[i - 1] << "\t" << i + 8 << '\t' << i + 9 << endl;
				deformfile << deformations << endl;
				centroidfile << centroids << endl;
			}

			i = i + 3;
		}
	}

	sizefile.close();
	deformfile.close();
	centroidfile.close();

	//waitKey(0);
}




//
// Get median of intensity
double Cyto::GetMedian(double daArray[], int iSize) {
	// Allocate an array of the same size and sort it.
	double* dpSorted = new double[iSize];
	for (int i = 0; i < iSize; ++i) {
		dpSorted[i] = daArray[i];
	}
	for (int i = iSize - 1; i > 0; --i) {
		for (int j = 0; j < i; ++j) {
			if (dpSorted[j] > dpSorted[j + 1]) {
				double dTemp = dpSorted[j];
				dpSorted[j] = dpSorted[j + 1];
				dpSorted[j + 1] = dTemp;
			}
		}
	}

	// Middle or average of middle values in the sorted array.
	double dMedian = 0.0;
	if ((iSize % 2) == 0) {
		dMedian = (dpSorted[iSize / 2] + dpSorted[(iSize / 2) - 1]) / 2.0;
	}
	else {
		dMedian = dpSorted[iSize / 2];
	}
	delete[] dpSorted;
	return dMedian;
}



//
double Cyto::Initial_Diameter(int i)
{
	/// Variables
	Mat dst_crop;
	Mat dst_binary;
	Mat dst_morph;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	IH imgHeader;
	IMRANGE imrange;
	//vector<unsigned char> cineData(imageH * imageW);
	
	/// Read images
	imrange.First = i;
	imrange.Cnt = 1;
	PhGetCineImage(cineHandle, (PIMRANGE)&imrange, m_pImageBuffer, imgSizeInBytes, (PIH)&imgHeader);
	Mat image = Mat(imageH, imageW, CV_8U, m_pImageBuffer);
	//imshow("imOrig", image);
	Mat imcrop(image, Rect(X0_dia, Y0_dia, Xlength_dia, Ylength_dia));  // crop image (0, 0, 35, 26)
	//imshow("imCrop", imcrop);

	//int morph_operator = 1;
	//int morph_elem = 2;
	//int morph_size = 5;

	int j = 0;
	int k = 0;
	double area = 0.0;
	double area_temp = 0.0;
	double dia_temp = 0.0;
	double equi_diameter = 0.0;

	double lda1 = 0.0;
	double lda2 = 0.0;
	double mup20 = 0.0;
	double mup02 = 0.0;
	double mup11 = 0.0;
	double Ecc = 0.0;
	///

	/*Initial Diameter*/
	///
	// Image Processing
	medianBlur(imcrop, dst_crop, Kernel_Size_dia);
	adaptiveThreshold(dst_crop, dst_binary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, Threshold1, Threshold2); //(7,3)
	dst_morph = Morphology_Operations(dst_binary, Morph_Operator_dia, Morph_Elem_dia, Morph_Size_dia);

	// For test
	//imshow("sizeO", dst_morph);
	
	/*
    Scalar intensity1 = dst_morph.at<uchar>(14, X0_dia);
	Scalar intensity2 = dst_morph.at<uchar>(15, X0_dia);
	Scalar intensity3 = dst_morph.at<uchar>(16, X0_dia);

	
	if (intensity1.val[0] || intensity2.val[0] || intensity3.val[0])
	{
		//cout << intensity1.val[0] << ", " << intensity2.val[0] << "," << intensity3.val[0] << endl;
		return 0; // cell on the boundary
	}
	*/
	// This part should put before "findContours", because "findContours" will modify "dst_morph"
	Scalar intensity1 = dst_morph.at<uchar>(14, X0_dia);
	Scalar intensity2 = dst_morph.at<uchar>(15, X0_dia);
	Scalar intensity3 = dst_morph.at<uchar>(16, X0_dia);
	Scalar intensity4 = dst_morph.at<uchar>(14, X0_dia + Xlength_dia - 1);
	Scalar intensity5 = dst_morph.at<uchar>(15, X0_dia + Xlength_dia - 1);
	Scalar intensity6 = dst_morph.at<uchar>(16, X0_dia + Xlength_dia - 1);


	if (intensity1.val[0] || intensity2.val[0] || intensity3.val[0] || intensity4.val[0] || intensity5.val[0] || intensity6.val[0])
	{
		// Cell on right boundary
		return 0;
	}


	// Find contours
	findContours(dst_morph, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	//cout << "contour = " << contours.size() << endl;

	if (contours.size() == 0)
	{
		return 0;
	}

	
	//  Get Max area
	for (int j_temp = 0; j_temp < contours.size(); j_temp++)
	{
		area_temp = contourArea(contours[j_temp]);
		dia_temp = sqrt(4 * area_temp / PI) / 0.56;
		//cout << "dia_temp=" << dia_temp << endl;
		if (dia_temp > 5)
		{
			k = k + 1;
			if (k > 1)
			{
				//cout << "multiple cell" << endl;
				return 0;
			}
		}
		if (area_temp > area)
		{
			area = area_temp;
			j = j_temp;
		}
	}



	// Delete objects on boundary
	
	for (int m = 0; m < contours[j].size(); m++)
	{
		if (contours[j][m].y < (Y0_dia + 3) || contours[j][m].y > (Y0_dia + Ylength_dia - 3)) // (1, 35)
		{
			return 0;
		}
	}
	

	// Get the moments
	vector<Moments> mu(1);
	vector<Point2f> mc(1);
	mu[0] = moments(contours[j], true);

	// Get the mass centers:
	mc[0] = Point2f(mu[0].m10 / mu[0].m00, mu[0].m01 / mu[0].m00);


	// Get eigenvalues
	mup20 = mu[0].m20 / mu[0].m00 - pow(mc[0].x, 2);
	mup02 = mu[0].m02 / mu[0].m00 - pow(mc[0].y, 2);
	mup11 = mu[0].m11 / mu[0].m00 - mc[0].x * mc[0].y;
	lda1 = (mup20 + mup02) / 2 + sqrt(4 * pow(mup11, 2) + pow((mup20 - mup02), 2)) / 2;
	lda2 = (mup20 + mup02) / 2 - sqrt(4 * pow(mup11, 2) + pow((mup20 - mup02), 2)) / 2;

	// Get eccentricity
	Ecc = sqrt(1 - lda2 / lda1);
	//cout << "ECC = " << Ecc << endl;
	// Get quivalent diameter
	equi_diameter = sqrt(4 * area / PI) / 0.56;

	if (equi_diameter < 6) //10
	{
		return 0;
	}

	if (Ecc > 0.75)
	{
		//cout << "Ecc= " << Ecc << endl;
		return 0;
	}

	// Store initial dimeter
	//cout << equi_diameter << endl;
	return equi_diameter;
	///
}



/* @Rule out close cells */
double Cyto::Detect(int i)
{
	/// Variables
	Mat dst_crop;
	Mat dst_binary;
	Mat dst_morph;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	IH imgHeader;
	IMRANGE imrange;
	//vector<unsigned char> cineData(imageH * imageW);

	/// Read images
	imrange.First = i;
	imrange.Cnt = 1;
	PhGetCineImage(cineHandle, (PIMRANGE)&imrange, m_pImageBuffer, imgSizeInBytes, (PIH)&imgHeader);
	Mat image = Mat(imageH, imageW, CV_8U, m_pImageBuffer);
	//imshow("imOrig1", image);
	Mat imcrop(image, Rect(X0_dia, Y0_dia, Xlength_dia, Ylength_dia));  // crop image (0, 0, 35, 26)
	//imshow("imCrop1", imcrop);

	double area1 = 0.0;
	double area2= 0.0;
	double equi_diameter1 = 0.0;
	double equi_diameter2 = 0.0;
	///

	/*Detect close cells*/
	///
	// Image Processing
	medianBlur(imcrop, dst_crop, 5);
	adaptiveThreshold(dst_crop, dst_binary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 7, Threshold2); //(7,3)
	dst_morph = Morphology_Operations(dst_binary, Morph_Operator_dia, Morph_Elem_dia, Morph_Size_dia);

	// For test
	//imshow("size1", dst_morph);

	// This part should put before "findContours", because "findContours" will modify "dst_morph"
	Scalar intensity1 = dst_morph.at<uchar>(14, X0_dia);
	Scalar intensity2 = dst_morph.at<uchar>(15, X0_dia);
	Scalar intensity3 = dst_morph.at<uchar>(16, X0_dia);
	Scalar intensity4 = dst_morph.at<uchar>(14, X0_dia + Xlength_dia - 1);
	Scalar intensity5 = dst_morph.at<uchar>(15, X0_dia + Xlength_dia - 1);
	Scalar intensity6 = dst_morph.at<uchar>(16, X0_dia + Xlength_dia - 1);

	
	
	// Find contours
	findContours(dst_morph, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	if (contours.size() == 0)
	{
		return 0;   // No cells
	}

	//cout << "i= " << i << ",   intensity = " << intensity2.val[0] << "  ,contour= " << contours.size() << endl;
	
	for (int j = 0; j < contours.size(); j++)
	{
		area1 = contourArea(contours[j]);
		equi_diameter1 = equi_diameter1 + sqrt(4 * area1 / PI) / 0.56;
		//cout << equi_diameter2 << endl;
	}

	if (equi_diameter1 < 5)
	{
		return 0;
	}

	if (intensity1.val[0] || intensity2.val[0] || intensity3.val[0] || intensity4.val[0] || intensity5.val[0] || intensity6.val[0])
	{
		// Cell on right boundary
		for (int j = 0; j < contours.size(); j++)
		{
			area2 = contourArea(contours[j]);
			equi_diameter2 = equi_diameter2 + sqrt(4 * area2 / PI) / 0.56;
		}
		
		//cout << "equi_sum= " << equi_diameter2 << endl; //
		
		if (equi_diameter2 < 23)
		{
			return 0;
		}
		/*
		for (int m = 0; m < contours[0].size(); m++)
		{
			if (contours[0][m].x < (X0_dia + 3) || contours[0][m].x >(X0_dia + Xlength_dia - 3))  // if the cell is on the right boundary
			{
				if (equi_diameter < 20)
				{
					return 0;
				}
			}
		}
		*/
	}

	//cout << equi_diameter << endl;
	return 1;
	///
}



/* @Deformation */
double Cyto::Deformation(int i)
{
	/// Variables
	//Mat image;
	Mat dst_crop;
	Mat dst_binary;
	Mat dst_morph;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	IH imgHeader;
	IMRANGE imrange;
	//vector<unsigned char> cineData(imageH * imageW);
	
	/// Read images
	imrange.First = i;
	imrange.Cnt = 1;
	PhGetCineImage(cineHandle, (PIMRANGE)&imrange, m_pImageBuffer, imgSizeInBytes, (PIH)&imgHeader);
	Mat image = Mat(imageH, imageW, CV_8U, m_pImageBuffer);
	
	Mat imcrop(image, Rect(X0_def, Y0_def, Xlength_def, Ylength_def));
	Mat dimage;
	equalizeHist(imcrop, dimage);

	int ym = 15;  // window coordinate in y_middle
	const int msize_b = 6; // length of intensity to get median of background intensity
	const int msize_c = 3; // length of intensity to get median of cell intensity

	double I_b[msize_b] = { 0 };

	for (int i = 0; i < 6; ++i)
	{
		Scalar intensity = dimage.at<uchar>(ym, i);

		I_b[i] = intensity.val[0];
		//cout << I_b[i] << endl;
		//ifile << I_b[i] << endl;
	}

	//Cyto t1;
	//cout << "median= " << t1.GetMedian(I,msize_b) << endl;

	int backIntensity;  // backgroud intensity
	backIntensity = GetMedian(I_b, msize_b);

	// int cutIntensity = backIntensity * 1.25;
	int cutI = backIntensity * 0.3;
	int upI = backIntensity * 0.7;



	//X0_def, Y0_def, Xlength_def, Ylength_def
	int xr1 = 109;  // window coordinate in x_right
	int xL1 = 90;   // window coordinate in x_left
	//int ym = 15;  // window coordinate in y_middle
	int x1 = 100; // scan from left
	int x2 = 0; // scan from right

	double upX = 0; // position with upI
	double loX = 0; // postion with lower Intensity
	double cutX = 0; // position with cut Intensity

	double th_t = 0; // temp threshold to find boundary
	double th1 = 0; // large threshold to find boundary
	double th2 = 0; // small threshold to find boundary
	double I_c[msize_c] = { 0 }; // intensity to find cells

	for (int x = 0; x <= Xlength_def; x++)
	{
		I_c[0] = dimage.at<uchar>(ym, x);
		I_c[1] = dimage.at<uchar>(ym - 1, x);
		I_c[2] = dimage.at<uchar>(ym + 1, x);
		//Scalar intensity4 = image.at<uchar>(ym - 2, x);
		//Scalar intensity5 = image.at<uchar>(ym + 2, x);
		//Scalar intensity6 = image.at<uchar>(ym - 3, x);
		//Scalar intensity7 = image.at<uchar>(ym + 3, x);
		th_t = GetMedian(I_c, msize_c);
		//cout << cutIntensity << endl;
		//cout << "x = " << x << ", " << intensity1.val[0] << ", " << intensity2.val[0] << ", " << intensity3.val[0] << ", " << intensity4.val[0] << ", " << intensity5.val[0] << ", " << intensity6.val[0] << ", " << intensity7.val[0] << endl;

		if (th_t < cutI)
		{
			loX = x;
			th2 = th_t;

			I_c[0] = dimage.at<uchar>(ym, x - 1);
			I_c[1] = dimage.at<uchar>(ym - 1, x - 1);
			I_c[2] = dimage.at<uchar>(ym + 1, x - 1);
			th1 = GetMedian(I_c, msize_c);

			break;
		}

		//if (intensity1.val[0] > cutIntensity || intensity2.val[0] > cutIntensity || intensity3.val[0] > cutIntensity || intensity4.val[0] > cutIntensity ||
		//	intensity5.val[0] > cutIntensity || intensity6.val[0] > cutIntensity || intensity7.val[0] > cutIntensity)
		//{
		//	cout << "0" << endl;
		//}
	}

	//cout << "uX = " << loX - 1 << " ,th1 = " << th1 << endl;
	//cout << "lX = " << loX << " ,th2 = " << th2 << endl;

	double deltaX = 0; // delta x
	deltaX = (cutI - th2) / (th1 - th2);
	//cout << "deltaX = " << deltaX << endl;

	if ((th1 - th2) == 0)
	{
		return 0;
	}

	double realX = loX - deltaX; // real X
	//cout << "realX = " << realX << endl;

	int wall_x = 18; // wall position
	int xm = (wall_x + loX) / 2;
	//cout << "xm = " << xm << endl;


	double th_tY1 = 0.0; // temp threshold to find boundary
	double th1Y1 = 0.0; // large threshold to find boundary
	double th2Y1 = 0.0; // small threshold to find boundary

	double upY1 = 0.0; // position with upI
	double loY1 = 0.0; // postion with lower Intensity
	double cutY1 = 0.0; // position with cut Intensity

	for (int y = ym + 10; y >= ym; y--)
	{
		I_c[0] = dimage.at<uchar>(y, xm);
		I_c[1] = dimage.at<uchar>(y, xm + 1);
		I_c[2] = dimage.at<uchar>(y, xm + 2);
		//Scalar intensity4 = image.at<uchar>(ym - 2, x);
		//Scalar intensity5 = image.at<uchar>(ym + 2, x);
		//Scalar intensity6 = image.at<uchar>(ym - 3, x);
		//Scalar intensity7 = image.at<uchar>(ym + 3, x);
		th_tY1 = GetMedian(I_c, msize_c);
		//cout << cutIntensity << endl;
		//cout << "x = " << x << ", " << intensity1.val[0] << ", " << intensity2.val[0] << ", " << intensity3.val[0] << ", " << intensity4.val[0] << ", " << intensity5.val[0] << ", " << intensity6.val[0] << ", " << intensity7.val[0] << endl;

		if (th_tY1 < cutI)
		{
			loY1 = y;
			th2Y1 = th_tY1;

			I_c[0] = dimage.at<uchar>(y + 1, xm);
			I_c[1] = dimage.at<uchar>(y + 1, xm + 1);
			I_c[2] = dimage.at<uchar>(y + 1, xm - 1);
			th1Y1 = GetMedian(I_c, msize_c);

			break;
		}
	}

	//cout << "uY = " << loY1 + 1 << " ,th1 = " << th1Y1 << endl;
	//cout << "lY = " << loY1 << " ,th2 = " << th2Y1 << endl;

	double deltaY1 = 0.0; // delta y
	deltaY1 = (cutI - th2Y1) / (th1Y1 - th2Y1);
	//cout << "deltaY1 = " << deltaY1 << endl;

	if ((th1Y1 - th2Y1) == 0)
	{
		return 0;
	}

	double realY1 = loY1 + deltaY1; // real Y
	//cout << "realY1 = " << realY1 << endl;


	double th_tY2 = 0; // temp threshold to find boundary
	double th1Y2 = 0; // large threshold to find boundary
	double th2Y2 = 0; // small threshold to find boundary

	double upY2 = 0; // position with upI
	double loY2 = 0; // postion with lower Intensity
	double cutY2 = 0; // position with cut Intensity

	for (int y = ym - 10; y <= ym; y++)
	{
		I_c[0] = dimage.at<uchar>(y, xm);
		I_c[1] = dimage.at<uchar>(y, xm + 1);
		I_c[2] = dimage.at<uchar>(y, xm + 2);
		//Scalar intensity4 = image.at<uchar>(ym - 2, x);
		//Scalar intensity5 = image.at<uchar>(ym + 2, x);
		//Scalar intensity6 = image.at<uchar>(ym - 3, x);
		//Scalar intensity7 = image.at<uchar>(ym + 3, x);
		th_tY2 = GetMedian(I_c, msize_c);
		//cout << cutIntensity << endl;
		//cout << "x = " << x << ", " << intensity1.val[0] << ", " << intensity2.val[0] << ", " << intensity3.val[0] << ", " << intensity4.val[0] << ", " << intensity5.val[0] << ", " << intensity6.val[0] << ", " << intensity7.val[0] << endl;

		if (th_tY2 < cutI)
		{
			loY2 = y;
			th2Y2 = th_tY2;

			I_c[0] = dimage.at<uchar>(y - 1, xm);
			I_c[1] = dimage.at<uchar>(y - 1, xm + 1);
			I_c[2] = dimage.at<uchar>(y - 1, xm - 1);
			th1Y2 = GetMedian(I_c, msize_c);

			break;
		}

		//if (intensity1.val[0] > cutIntensity || intensity2.val[0] > cutIntensity || intensity3.val[0] > cutIntensity || intensity4.val[0] > cutIntensity ||
		//	intensity5.val[0] > cutIntensity || intensity6.val[0] > cutIntensity || intensity7.val[0] > cutIntensity)
		//{
		//	cout << "0" << endl;
		//}
	}

	//cout << "uY2 = " << loY2 + 1 << " ,th1 = " << th1Y2 << endl;
	//cout << "lY2 = " << loY2 << " ,th2 = " << th2Y2 << endl;

	double deltaY2 = 0; // delta x
	deltaY2 = (cutI - th2Y2) / (th1Y2 - th2Y2);
	
	if ((th1Y2 - th2Y2) == 0)
	{
		return 0;
	}

	//cout << "deltaY2 = " << deltaY2 << endl;

	double realY2 = loY2 - deltaY2; // real X
	//cout << "realY2 = " << realY2 << endl;

	if ((realY1 - realY2)< 1 || (wall_x - realX) < 1)
	{
		return 0;
	}
	
	double deform = 0.0;
	deform = (realY1 - realY2) / (wall_x - realX);

	//deform = major / minor;
	//cout << "deformability = " << deform << endl;   //

	if (deform > 3 || deform < 1)
	{

		return 0;
	}

	//cout << deform << endl;
	return deform;
	///
	
	///
}



/// Deformation Detect

int Cyto::Detect_Def(int i)
{
	IH imgHeader;
	IMRANGE imrange;
	//vector<unsigned char> cineData(imageH * imageW);

	/// Read images
	imrange.First = i;
	imrange.Cnt = 1;
	PhGetCineImage(cineHandle, (PIMRANGE)&imrange, m_pImageBuffer, imgSizeInBytes, (PIH)&imgHeader);
	Mat image = Mat(imageH, imageW, CV_8U, m_pImageBuffer);

	int backIntensity = 68;  // backgroud intensity (MMM)
	int xr = 110;  // window coordinate in x_right
	
	int cutIntensity = backIntensity * 1.25;   

	int xL = 90;   // window coordinate in x_left
	int ym = 15;  // window coordinate in y_middle
	int x1 = 100; // scan from left
	int x2 = 0; // scan from right

	for (int x = xL; x <= xr; x++)
	{
		Scalar intensity1 = image.at<uchar>(ym, x);
		Scalar intensity2 = image.at<uchar>(ym - 1, x);
		Scalar intensity3 = image.at<uchar>(ym + 1, x);
		Scalar intensity4 = image.at<uchar>(ym - 2, x);
		Scalar intensity5 = image.at<uchar>(ym + 2, x);
		Scalar intensity6 = image.at<uchar>(ym - 3, x);
		Scalar intensity7 = image.at<uchar>(ym + 3, x);

		//cout << intensity1.val[0] << intensity2.val[0] << intensity3.val[0] << intensity4.val[0] << intensity5.val[0] << intensity6.val[0] << intensity7.val[0] << endl;

		if (intensity1.val[0] > cutIntensity || intensity2.val[0] > cutIntensity || intensity3.val[0] > cutIntensity || intensity4.val[0] > cutIntensity ||
			intensity5.val[0] > cutIntensity || intensity6.val[0] > cutIntensity || intensity7.val[0] > cutIntensity)
		{
			return 0;
		}
	}

	return 1;
}



//
double Cyto::Centroid(int i)
{
	/// Variables
	Mat dst_crop;
	Mat dst_binary;
	Mat dst_morph;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	IH imgHeader;
	IMRANGE imrange;
	//vector<unsigned char> cineData(imageH * imageW);

	/// Read images
	imrange.First = i;
	imrange.Cnt = 1;
	PhGetCineImage(cineHandle, (PIMRANGE)&imrange, m_pImageBuffer, imgSizeInBytes, (PIH)&imgHeader);
	Mat image = Mat(imageH, imageW, CV_8U, m_pImageBuffer);
	//imshow("imOrig", image);
	Mat imcrop(image, Rect(X0_dia, Y0_dia, Xlength_dia, Ylength_dia));  // crop image (0, 0, 35, 26)
	//imshow("imCrop", imcrop);

	//int morph_operator = 1;
	//int morph_elem = 2;
	//int morph_size = 5;

	int j = 0;
	int k = 0;
	double area = 0.0;
	double area_temp = 0.0;
	double dia_temp = 0.0;
	double centroid = 0.0;

	double lda1 = 0.0;
	double lda2 = 0.0;
	double mup20 = 0.0;
	double mup02 = 0.0;
	double mup11 = 0.0;
	double Ecc = 0.0;
	///

	/*Initial Diameter*/
	///
	// Image Processing
	medianBlur(imcrop, dst_crop, Kernel_Size_dia);
	adaptiveThreshold(dst_crop, dst_binary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, Threshold1, Threshold2); //(7,3)
	dst_morph = Morphology_Operations(dst_binary, Morph_Operator_dia, Morph_Elem_dia, Morph_Size_dia);

	// Find contours
	findContours(dst_morph, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	//cout << "contour = " << contours.size() << endl;

	//  Get Max area
	for (int j_temp = 0; j_temp < contours.size(); j_temp++)
	{
		area_temp = contourArea(contours[j_temp]);
		dia_temp = sqrt(4 * area_temp / PI) / 0.56;
		//cout << "dia_temp=" << dia_temp << endl;
		if (dia_temp > 5)
		{
			k = k + 1;
			if (k > 1)
			{
				//cout << "multiple cell" << endl;
				return 0;
			}
		}
		if (area_temp > area)
		{
			area = area_temp;
			j = j_temp;
		}
	}


	// Get the moments
	vector<Moments> mu(1);
	vector<Point2f> mc(1);
	mu[0] = moments(contours[j], true);

	// Get the mass centers:
	mc[0] = Point2f(mu[0].m10 / mu[0].m00, mu[0].m01 / mu[0].m00);


    // Get centroid
	centroid = mc[0].y;

	return centroid;
	///
}





/*@function Morphology_Operations*/
Mat Cyto::Morphology_Operations(Mat dst_binary, int morph_operator, int morph_elem, int morph_size)
{
	Mat dst_morph;

	/// Since MORPH_OP : 2,3,4,5 and 6
	int operation = morph_operator + 2;

	Mat element = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));

	/// Apply the specified morphology operation
	morphologyEx(dst_binary, dst_morph, operation, element);

	// imshow("Morph_Op", dst_morph);         // For test

	return dst_morph;
}

