#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() { //BOTH

  x_ = F_ * x_; //u = 0
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;

}

void KalmanFilter::Update(const VectorXd &z) { //LASER

  VectorXd y = z - H_ * x_;
  MatrixXd Ht = H_.transpose(); 
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K =  (P_ * Ht) * Si;

  // New state
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;

}

void KalmanFilter::UpdateEKF(const VectorXd &z) { //RADAR

  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  if(fabs(px) < 0.001) {
    px = 0.001;
  }

  float rho = sqrt(pow(px,2) + pow(py,2));

  if (fabs(rho) < 0.001){
    rho = 0.001;
  }
 
  float phi = atan2(py,px);
  float ro_dot = (px*vx + py*vy)/rho;
  VectorXd z_pred = VectorXd(3);
  z_pred << rho, phi, ro_dot;


  // Update
  VectorXd y = z - z_pred;
 
  if(fabs(y[1]) > M_PI){
    y[1] = atan2(sin(y[1]), cos(y[1]));
  }

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K =  (P_ * Ht) * Si;

  // New state
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;

}


