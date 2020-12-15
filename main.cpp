#include <QApplication>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);
//---------------------------------------------------------------------------------------------------- 

  Mat pix_offset_matrix = Mat::zeros( 12, 2, CV_32SC3 );

  ifstream dataset_file( "/home/skr/qt_projects/lab_3_ida/RGB.txt" );

// read data from file

  int line_num = 0;

  while( dataset_file )
  {
    string cur_str_line;
    getline( dataset_file, cur_str_line );
    const char *cur_char_line = cur_str_line.c_str();

    char x_coor [32];
    char y_coor [32];

    sscanf( cur_char_line, "%[^,] %*[,] %s", x_coor, y_coor );

    int x_coor_num = stoul( x_coor, nullptr );
    int y_coor_num = stoul( y_coor, nullptr );

    if(line_num <= 11)
    {
      pix_offset_matrix.at<Vec3i>(line_num,0)[2] = x_coor_num;
      pix_offset_matrix.at<Vec3i>(line_num,1)[2] = y_coor_num;
    }
    else if(line_num <= 23)
    {
      pix_offset_matrix.at<Vec3i>(line_num-12,0)[1] = x_coor_num;
      pix_offset_matrix.at<Vec3i>(line_num-12,1)[1] = y_coor_num;
    }
    else if(line_num <= 35)
    {
      pix_offset_matrix.at<Vec3i>(line_num-24,0)[0] = x_coor_num;
      pix_offset_matrix.at<Vec3i>(line_num-24,1)[0] = y_coor_num;
    }

    line_num++;
  }

  for(int i=0; i < pix_offset_matrix.rows; i++)
  {
    cout << pix_offset_matrix.at<Vec3i>(i,0)[2] << " " <<pix_offset_matrix.at<Vec3i>(i,0)[1] << " " << pix_offset_matrix.at<Vec3i>(i,0)[0] << endl; 
  }

//---------------------------------------------------------------------------------------------------- 
  return a.exec();
}
