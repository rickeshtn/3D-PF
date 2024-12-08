#ifndef UKF_CTRV_H
#define UKF_CTRV_H

#include <Eigen/Dense>
#include <ros/ros.h>

class UKF_CTRV {
public:
    UKF_CTRV();
    void initialize(double px, double py, double v, double yaw, double yaw_rate);
    bool isInitialized() const { return initialized_; }

    void predict(double dt);
    void updatePoseMeasurement(double px_meas, double py_meas, double yaw_meas);

    double px() const { return x_(0); }
    double py() const { return x_(1); }
    double v()  const { return x_(2); }
    double yaw() const { return x_(3); }
    double yawRate() const { return x_(4); }

    void setProcessNoise(double std_a, double std_yawdd);
    void setMeasurementNoise(double std_px, double std_py, double std_yaw);

private:
    bool initialized_;
    int n_x_;
    int n_aug_;
    double lambda_;

    Eigen::VectorXd x_;  // state: [px, py, v, yaw, yaw_rate]
    Eigen::MatrixXd P_;  // state covariance
    Eigen::MatrixXd Xsig_pred_; // Predicted sigma points

    // Noise parameters
    double std_a_;
    double std_yawdd_;
    double std_px_meas_;
    double std_py_meas_;
    double std_yaw_meas_;

    Eigen::VectorXd weights_;

    Eigen::MatrixXd generateSigmaPoints();
    Eigen::MatrixXd predictSigmaPoints(const Eigen::MatrixXd &Xsig_aug, double dt);
    void predictMeanAndCovariance(const Eigen::MatrixXd &Xsig_pred);
    void normalizeAngle(double &angle);
};

#endif
