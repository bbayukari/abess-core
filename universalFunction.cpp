#ifdef R_BUILD // this file is only used wich Rcpp

// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(StanHeaders)]]
#include <stan/math.hpp>
#include <Rcpp.h>
#include "UniversalData.h"

// [[Rcpp::plugins(cpp14)]]
using namespace Eigen;
// TODO: Define a global variable ExternData and no longer pass data pointer
struct ExternData{
/************************************** User defined **************************************/
    MatrixXd x;
    VectorXd y;
    ExternData(MatrixXd x, VectorXd y) {
        this->x = x;
        this->y = y;
    }
/************************************** User defined **************************************/
};


class AutoDiffFunction :public UniversalData {
private:
    VectorXd const *effective_para_ptr;
public:
    AutoDiffFunction(const UniversalData& universal_data, VectorXd const *effective_para_ptr)
        :UniversalData(universal_data), effective_para_ptr(effective_para_ptr)
    {
        // assert(this->effective_para.size() == this->effective_para_index.size());
        if (this->compute_para_index_ptr == NULL) {
            this->compute_para_index_ptr = &this->effective_para_index;
        }
    }
    template <typename T> 
    T operator()(const VectorX<T>& compute_para) const {
        ExternData*  data = static_cast<ExternData*>(this->data);
        VectorX<T> para(this->model_size);
        for (int i = 0; i < para.size(); i++) {
            para[i] = T(0.0);
        }
        for (int i = 0; i < this->effective_para_index.size(); i++) {
            para[this->effective_para_index[i]] = T((*this->effective_para_ptr)[i]);
        }
        for (int i = 0; i < this->compute_para_index_ptr->size(); i++) {
            para[(*this->compute_para_index_ptr)[i]] = compute_para[i];
        }
/************************************** User defined **************************************/
        return (data->x * para - data->y).cwiseAbs2().sum();
/************************************** User defined **************************************/
    }
};

double comput_value(const VectorXd& effective_para, const UniversalData& universal_data) {
    AutoDiffFunction function(universal_data, &effective_para);
    VectorXd compute_para;
    function.get_compute_para(effective_para, compute_para);
    return function(compute_para);
}

double compute_gradient(const VectorXd& effective_para, const UniversalData& universal_data, VectorXd* gradient) {
    double value = 0;
    AutoDiffFunction function(universal_data, &effective_para);
    VectorXd compute_para;
    function.get_compute_para(effective_para, compute_para);
    stan::math::gradient(function, compute_para, value, *gradient);
    return value;
}

double compute_hessian(const VectorXd& effective_para, const UniversalData& universal_data, VectorXd* gradient, MatrixXd* hessian) {
    double value = 0;  
    AutoDiffFunction function(universal_data, &effective_para);
    VectorXd compute_para;
    function.get_compute_para(effective_para, compute_para);
    if (!gradient) {
        auto grad = VectorXd(compute_para.size());
        gradient = &grad;
    }
    stan::math::hessian(function, compute_para, value, *gradient, *hessian);
    return value;
}

double function(const VectorXd& effective_para, const UniversalData& universal_data, VectorXd* gradient, MatrixXd* hessian) {
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
    MatrixXd x, VectorXd y
    /************************************** User defined **************************************/
) {
    ExternData* ptr = new ExternData(x,y);
    return Rcpp::XPtr<void*>(new (void*) ptr);
}


#endif // R_BUILD