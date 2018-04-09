#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

	VectorXd rmse(4);
	rmse << 0,0,0,0;

	// Check that the estimation vector size is not zero
	// Check that the estimation vector size equals ground truth vector size

	if(estimations.size() != ground_truth.size() || estimations.size() == 0){
		cout << "Invalid estimation or ground_truth data" << endl;
		return rmse;
	}
	// Accumulate squared residuals
	for(int i=0; i < estimations.size(); ++i){
        
		VectorXd residual = estimations[i] - ground_truth[i];

		residual = residual.array()*residual.array();
		rmse += residual;
	}

	// Calculate the mean
	rmse = rmse/estimations.size();
	
	// Calculate the squared root
	rmse = rmse.array().sqrt();

	// Return the result
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
	MatrixXd Hj(3,4);
	// Recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);
	
	// Denominators
    float d1 = pow(px,2) + pow(py,2);
    float d2 = sqrt(d1);
    float d3 = pow(d1, 1.5);
    
	// Check division by zero
	if (fabs(d1) <0.00001){
		cout << "CalculateJacobian () - Error - Division by Zero" << endl;
	    return Hj;

	}
	
	// Jacobian matrix
	Hj << (px/d2), (py/d2), 0, 0,
	       -(py/d1), (px/d1), 0, 0,
	       (py*(vx*py-vy*px))/d3, (px*(vy*px-vx*py))/d3, px/d2, py/d2;

	return Hj;
}

