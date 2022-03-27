// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(StanHeaders)]]
#include <stan/math.hpp>
#include <Rcpp.h>
#include "UniversalData.h"

// [[Rcpp::plugins(cpp14)]]


struct ExternData{
/************************************** User defined **************************************/
    Eigen::MatrixXd x;
    Eigen::VectorXd y;
    ExternData(Eigen::MatrixXd x, Eigen::VectorXd y) {
        this->x = x;
        this->y = y;
    }
/************************************** User defined **************************************/
};


class AutoDiffFunction :public UniversalData {
private:
    const Eigen::VectorXd effective_para;
public:
    AutoDiffFunction(const UniversalData& universal_data, const Eigen::VectorXd& effective_para):UniversalData(universal_data), effective_para(effective_para) {
        // assert(this->effective_para.size() == this->effective_para_index.size());
        if (this->compute_para_index.size() == 0) {
            this->compute_para_index = this->effective_para_index;
        }
        this->data = (ExternData*)this->data;
    }
    template <typename T> 
    T operator()(const Eigen::VectorX<T>& compute_para) const {
        // assert(this->compute_para_index.size() == compute_para.size());
        Eigen::VectorX<T> para(this->dim);
        for (int i = 0; i < para.size(); i++) {
            para[i] = T(0.0);
        }
        for (int i = 0; i < this->effective_para_index.size(); i++) {
            para[this->effective_para_index[i]] = T(effective_para[i]);
        }
        for (int i = 0; i < this->compute_para_index.size(); i++) {
            para[this->compute_para_index[i]] = compute_para[i];
        }
/************************************** User defined **************************************/
        return (data->x * para - data->y).cwiseAbs2().sum();
/************************************** User defined **************************************/
    }
};

double comput_value(const Eigen::VectorXd& effective_para, const UniversalData& universal_data) {
    AutoDiffFunction function(universal_data, effective_para);
    Eigen::VectorXd compute_para;
    function.get_compute_para(effective_para, compute_para);
    return function(compute_para);
}

double compute_gradient(const Eigen::VectorXd& effective_para, const UniversalData& universal_data, Eigen::VectorXd* gradient) {
    double value = 0;
    AutoDiffFunction function(universal_data, effective_para);
    Eigen::VectorXd compute_para;
    function.get_compute_para(effective_para, compute_para);
    stan::math::gradient(function, compute_para, value, *gradient);
    return value;
}

double compute_hessian(const Eigen::VectorXd& effective_para, const UniversalData& universal_data, Eigen::VectorXd* gradient, Eigen::MatrixXd* hessian) {
    double value = 0;  
    AutoDiffFunction function(universal_data, effective_para);
    Eigen::VectorXd compute_para;
    function.get_compute_para(effective_para, compute_para);
    if (!gradient) {
        auto grad = Eigen::VectorXd(compute_para.size());
        gradient = &grad;
    }
    stan::math::hessian(function, compute_para, value, *gradient, *hessian);
    return value;
}

double function(const Eigen::VectorXd& effective_para, const UniversalData& universal_data, Eigen::VectorXd* gradient, Eigen::MatrixXd* hessian) {
    if (hessian) {
        return compute_hessian(effective_para, universal_data, gradient, hessian);
    }
    if (gradient) {
        return compute_gradient(effective_para, universal_data, gradient);
    }
    return comput_value(effective_para, universal_data);
}

// [[Rcpp::export]]
Rcpp::XPtr<universal_function> get_universal_function() {
    return Rcpp::XPtr<universal_function>(new universal_function(&function));
}

// [[Rcpp::export]]
Rcpp::XPtr<void*> get_extern_data(
    /************************************** User defined **************************************/
    Eigen::MatrixXd x, Eigen::VectorXd y
    /************************************** User defined **************************************/
) {
    ExternData* ptr = new ExternData(x,y);
    return Rcpp::XPtr<void*>(new (void*) ptr);
}
