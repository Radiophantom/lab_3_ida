#include <QApplication>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <vector>

using namespace cv;
using namespace std;

const int point_num = 64;

void calc_A_func( Mat& reference_channel, Mat& calc_channel, Mat& Ax_matrix, Mat& Ay_matrix )
{
  Mat x_dif_mat = calc_channel.col(0) - reference_channel.col(0) ;
  x_dif_mat.convertTo( x_dif_mat, CV_64F );

  Mat y_dif_mat = calc_channel.col(1) - reference_channel.col(1);
  y_dif_mat.convertTo( y_dif_mat, CV_64F );
  
  Mat F_matrix = Mat::zeros( point_num, 10, CV_64F );
  for( int i=0; i<F_matrix.rows; i++ )
  {
    F_matrix.at<double>(i,0) = 1;
    F_matrix.at<double>(i,1) = reference_channel.at<int>(i,0);
    F_matrix.at<double>(i,2) = reference_channel.at<int>(i,1);
    F_matrix.at<double>(i,3) = reference_channel.at<int>(i,0) * reference_channel.at<int>(i,1);
    F_matrix.at<double>(i,4) = reference_channel.at<int>(i,0) * reference_channel.at<int>(i,0);
    F_matrix.at<double>(i,5) = reference_channel.at<int>(i,1) * reference_channel.at<int>(i,1);
    F_matrix.at<double>(i,6) = reference_channel.at<int>(i,0) * reference_channel.at<int>(i,0) * reference_channel.at<int>(i,1);
    F_matrix.at<double>(i,7) = reference_channel.at<int>(i,1) * reference_channel.at<int>(i,1) * reference_channel.at<int>(i,0);
    F_matrix.at<double>(i,8) = reference_channel.at<int>(i,0) * reference_channel.at<int>(i,0) * reference_channel.at<int>(i,0);
    F_matrix.at<double>(i,9) = reference_channel.at<int>(i,1) * reference_channel.at<int>(i,1) * reference_channel.at<int>(i,1);
  }
  
  Mat F_matrix_trans = F_matrix.t();
  Mat F_mult = (F_matrix_trans * F_matrix);
  invert( F_mult, F_mult );
  F_mult *= F_matrix_trans;

  Ax_matrix = F_mult * x_dif_mat;
  Ay_matrix = F_mult * y_dif_mat;

}

void calc_coordinate( int& x, int& y, Mat& Ax, Mat& Ay )
{
  double x_offset = 0;
  double y_offset = 0;
  
  x_offset =  Ax.at<double>(0,0) + 
              Ax.at<double>(1,0)*x + 
              Ax.at<double>(2,0)*y + 
              Ax.at<double>(3,0)*x*y + 
              Ax.at<double>(4,0)*x*x + 
              Ax.at<double>(5,0)*y*y + 
              Ax.at<double>(6,0)*x*x*y + 
              Ax.at<double>(7,0)*y*y*x + 
              Ax.at<double>(8,0)*x*x*x + 
              Ax.at<double>(9,0)*y*y*y;

  y_offset =  Ay.at<double>(0,0) + 
              Ay.at<double>(1,0)*x + 
              Ay.at<double>(2,0)*y + 
              Ay.at<double>(3,0)*x*y + 
              Ay.at<double>(4,0)*x*x + 
              Ay.at<double>(5,0)*y*y + 
              Ay.at<double>(6,0)*x*x*y + 
              Ay.at<double>(7,0)*y*y*x + 
              Ay.at<double>(8,0)*x*x*x + 
              Ay.at<double>(9,0)*y*y*y;

  x_offset = round( x_offset );
  y_offset = round( y_offset );

  x += x_offset;
  y += y_offset;
}


void correct_channel_func( Mat& orig_channel, Mat& corrected_channel, Mat& Ax_mat, Mat& Ay_mat )
{
  for( int i=0; i<corrected_channel.rows; i++ )
  {
    for( int j=0; j<corrected_channel.cols; j++ )
    {
      int x = j;
      int y = i;
      calc_coordinate( x, y, Ax_mat, Ay_mat );
      corrected_channel.at<unsigned char>(i,j) = orig_channel.at<unsigned char>(y,x);
    }
  }
}

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);
//---------------------------------------------------------------------------------------------------- 

  Mat r_channel_pix = Mat::zeros( point_num, 2, CV_32S );
  Mat g_channel_pix = Mat::zeros( point_num, 2, CV_32S );
  Mat b_channel_pix = Mat::zeros( point_num, 2, CV_32S );

  ifstream dataset_file( "/home/skr/qt_projects/lab_3_ida/table_point.txt" );

// read data from file

  int line_num = 0;

  while( dataset_file )
  {
    string cur_str_line;
    getline( dataset_file, cur_str_line );
    const char *cur_char_line = cur_str_line.c_str();

    char x_coor [32];
    char y_coor [32];

    sscanf( cur_char_line, "%[^,] %*[,] %s", y_coor, x_coor );

    if(line_num <= point_num-1)
    {
      r_channel_pix.at<int>(line_num,0) = stoul( x_coor, nullptr );
      r_channel_pix.at<int>(line_num,1) = stoul( y_coor, nullptr );
    }
    else if(line_num <= point_num*2 - 1)
    {
      g_channel_pix.at<int>(line_num-point_num,0) = stoul( x_coor, nullptr );
      g_channel_pix.at<int>(line_num-point_num,1) = stoul( y_coor, nullptr );
    }
    else if(line_num <= point_num*3 - 1)
    {
      b_channel_pix.at<int>(line_num-point_num*2,0) = stoul( x_coor, nullptr );
      b_channel_pix.at<int>(line_num-point_num*2,1) = stoul( y_coor, nullptr );
    }

    line_num++;
  }
  
  Mat Ax_r_channel = Mat::zeros( point_num, 1, CV_64F ); Mat Ay_r_channel = Mat::zeros( point_num, 1, CV_64F );
  Mat Ax_b_channel = Mat::zeros( point_num, 1, CV_64F ); Mat Ay_b_channel = Mat::zeros( point_num, 1, CV_64F );

  calc_A_func( g_channel_pix, r_channel_pix, Ax_r_channel, Ay_r_channel );
  calc_A_func( g_channel_pix, b_channel_pix, Ax_b_channel, Ay_b_channel );
  
  Mat orig_img = imread("/home/skr/qt_projects/lab_3_ida/test_img.png");

  vector<Mat> sp_img;
  split(orig_img, sp_img);
  Mat R_channel = sp_img[2].clone();
  Mat G_channel = sp_img[1].clone();
  Mat B_channel = sp_img[0].clone();

  Mat R_channel_corrected = Mat::zeros( R_channel.size(), CV_8UC1 );
  Mat B_channel_corrected = Mat::zeros( B_channel.size(), CV_8UC1 );

  correct_channel_func( R_channel, R_channel_corrected, Ax_r_channel, Ay_r_channel );
  correct_channel_func( B_channel, B_channel_corrected, Ax_b_channel, Ay_b_channel );

  vector<Mat> mg_img;
  mg_img.push_back(B_channel_corrected);
  mg_img.push_back(G_channel);
  mg_img.push_back(R_channel_corrected);

  Mat cor_img = Mat::zeros( orig_img.size(), CV_8UC3 );
  merge(mg_img, cor_img);

  imshow("original image", orig_img);
  imshow("corrected image", cor_img);

  waitKey();

//---------------------------------------------------------------------------------------------------- 
  return a.exec();
}
