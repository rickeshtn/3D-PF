#include "ukf_ctrv.h"
#include <cmath>

UKF_CTRV::UKF_CTRV()
    : initialized_(false)
{
    n_x_ = 5;
    n_aug_ = n_x_ + 2;
    lambda_ = 3 - n_aug_;

    x_ = Eigen::VectorXd::Zero(n_x_);
    P_ = Eigen::MatrixXd::Identity(n_x_, n_x_);

    std_a_ = 1.0;
    std_yawdd_ = 0.5;
    std_px_meas_ = 0.5;
    std_py_meas_ = 0.5;
    std_yaw_meas_ = 0.5;

    weights_ = Eigen::VectorXd(2 * n_aug_ + 1);
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i < 2*n_aug_+1; i++) {
        weights_(i) = 0.5/(lambda_+n_aug_);
    }
}

void UKF_CTRV::initialize(double px, double py, double v, double yaw, double yaw_rate) {
    x_ << px, py, v, yaw, yaw_rate;
    P_ = Eigen::MatrixXd::Identity(n_x_, n_x_);
    initialized_ = true;
}

void UKF_CTRV::setProcessNoise(double std_a, double std_yawdd) {
    std_a_ = std_a;
    std_yawdd_ = std_yawdd;
}

void UKF_CTRV::setMeasurementNoise(double std_px, double std_py, double std_yaw) {
    std_px_meas_ = std_px;
    std_py_meas_ = std_py;
    std_yaw_meas_ = std_yaw;
}

void UKF_CTRV::normalizeAngle(double &angle) {
    while (angle > M_PI) angle -= 2.0*M_PI;
    while (angle < -M_PI) angle += 2.0*M_PI;
}

Eigen::MatrixXd UKF_CTRV::generateSigmaPoints() {
    Eigen::VectorXd x_aug = Eigen::VectorXd(n_aug_);
    x_aug.head(n_x_) = x_;
    x_aug(n_x_) = 0;
    x_aug(n_x_+1) = 0;

    Eigen::MatrixXd P_aug = Eigen::MatrixXd::Zero(n_aug_, n_aug_);
    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    P_aug(n_x_, n_x_) = std_a_*std_a_;
    P_aug(n_x_+1, n_x_+1) = std_yawdd_*std_yawdd_;

    Eigen::MatrixXd Xsig_aug = Eigen::MatrixXd(n_aug_, 2*n_aug_+1);
    Eigen::MatrixXd A = P_aug.llt().matrixL();

    Xsig_aug.col(0) = x_aug;
    double factor = sqrt(lambda_ + n_aug_);
    for (int i = 0; i < n_aug_; i++) {
        Xsig_aug.col(i+1)        = x_aug + factor * A.col(i);
        Xsig_aug.col(i+1+n_aug_) = x_aug - factor * A.col(i);
    }
    return Xsig_aug;
}

Eigen::MatrixXd UKF_CTRV::predictSigmaPoints(const Eigen::MatrixXd &Xsig_aug, double dt) {
    Eigen::MatrixXd Xsig_pred = Eigen::MatrixXd(n_x_, 2*n_aug_+1);

    for (int i = 0; i < 2*n_aug_+1; i++){
        double px = Xsig_aug(0,i);
        double py = Xsig_aug(1,i);
        double v  = Xsig_aug(2,i);
        double yaw = Xsig_aug(3,i);
        double yawd = Xsig_aug(4,i);
        double nu_a = Xsig_aug(5,i);
        double nu_yawdd = Xsig_aug(6,i);

        double px_p, py_p;

        if (fabs(yawd) > 1e-6) {
            px_p = px + v/yawd*(sin(yaw+yawd*dt) - sin(yaw));
            py_p = py + v/yawd*(-cos(yaw+yawd*dt) + cos(yaw));
        } else {
            px_p = px + v*dt*cos(yaw);
            py_p = py + v*dt*sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd*dt;
        double yawd_p = yawd;

        // Add process noise
        px_p += 0.5*nu_a*dt*dt*cos(yaw);
        py_p += 0.5*nu_a*dt*dt*sin(yaw);
        v_p  += nu_a*dt;
        yaw_p += 0.5*nu_yawdd*dt*dt;
        yawd_p+= nu_yawdd*dt;

        Xsig_pred(0,i) = px_p;
        Xsig_pred(1,i) = py_p;
        Xsig_pred(2,i) = v_p;
        Xsig_pred(3,i) = yaw_p;
        Xsig_pred(4,i) = yawd_p;
    }

    return Xsig_pred;
}

void UKF_CTRV::predictMeanAndCovariance(const Eigen::MatrixXd &Xsig_pred) {
    x_.fill(0.0);
    for (int i = 0; i < 2*n_aug_+1; i++) {
        x_ += weights_(i)*Xsig_pred.col(i);
    }

    P_.fill(0.0);
    for (int i = 0; i < 2*n_aug_+1; i++) {
        Eigen::VectorXd diff = Xsig_pred.col(i) - x_;
        normalizeAngle(diff(3));
        P_ += weights_(i)*diff*diff.transpose();
    }
}

void UKF_CTRV::predict(double dt) {
    Eigen::MatrixXd Xsig_aug = generateSigmaPoints();
    Xsig_pred_ = predictSigmaPoints(Xsig_aug, dt);
    predictMeanAndCovariance(Xsig_pred_);
}

void UKF_CTRV::updatePoseMeasurement(double px_meas, double py_meas, double yaw_meas) {
    int n_z = 3;

    Eigen::MatrixXd Zsig = Eigen::MatrixXd(n_z, 2*n_aug_+1);
    for (int i = 0; i < 2*n_aug_+1; i++) {
        double px = Xsig_pred_(0,i);
        double py = Xsig_pred_(1,i);
        double yaw = Xsig_pred_(3,i);
        normalizeAngle(yaw);
        Zsig(0,i) = px;
        Zsig(1,i) = py;
        Zsig(2,i) = yaw;
    }

    // mean predicted measurement
    Eigen::VectorXd z_pred = Eigen::VectorXd::Zero(n_z);
    for (int i = 0; i < 2*n_aug_+1; i++){
        z_pred += weights_(i)*Zsig.col(i);
    }

    // measurement covariance S
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(n_z, n_z);
    for (int i = 0; i < 2*n_aug_+1; i++){
        Eigen::VectorXd diff = Zsig.col(i) - z_pred;
        normalizeAngle(diff(2));
        S += weights_(i)*diff*diff.transpose();
    }

    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n_z,n_z);
    R(0,0) = std_px_meas_*std_px_meas_;
    R(1,1) = std_py_meas_*std_py_meas_;
    R(2,2) = std_yaw_meas_*std_yaw_meas_;
    S += R;

    // cross correlation Tc
    Eigen::MatrixXd Tc = Eigen::MatrixXd::Zero(n_x_, n_z);
    for (int i = 0; i < 2*n_aug_+1; i++){
        Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;
        normalizeAngle(x_diff(3));

        Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
        normalizeAngle(z_diff(2));

        Tc += weights_(i)*x_diff*z_diff.transpose();
    }

    Eigen::VectorXd z = Eigen::VectorXd(n_z);
    z << px_meas, py_meas, yaw_meas;
    Eigen::MatrixXd K = Tc * S.inverse();
    Eigen::VectorXd z_diff = z - z_pred;
    normalizeAngle(z_diff(2));

    x_ += K*z_diff;
    P_ -= K*S*K.transpose();
}
