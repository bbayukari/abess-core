code = "
// [[Rcpp::depends(abess)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(StanHeaders)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::plugins(cpp14)]]
#include <stan/math/rev/mat.hpp>
#include <stan/math/fwd/mat.hpp>// stuff from fwd/ must come first, then stuff from mix/ must come next
#include <stan/math/mix/mat.hpp>

#include <Rcpp.h>
#include <RcppEigen.h>
#include <abessUniversal.h>

using namespace Eigen;
using namespace stan::math; // necessary to get the type promotion correct

struct ExternData {
    /************************************* User defined **************************************/
    MatrixXd x;
    VectorXd y;
    /************************************** User defined **************************************/
} extern_data;

class AutoDiffFunction :public UniversalData {
private:
    VectorXd const* effective_para_ptr;
public:
    AutoDiffFunction(const UniversalData& universal_data, VectorXd const* effective_para_ptr)
        :UniversalData(universal_data), effective_para_ptr(effective_para_ptr)
    {
        if (this->compute_para_index_ptr == NULL) {
            this->compute_para_index_ptr = &this->effective_para_index;
        }
    }
    template <typename T>
    T operator()(const Matrix<T, -1, 1>& compute_para) const {
        Matrix<T, -1, 1> para(this->model_size);
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
        VectorXd one = VectorXd::Ones(extern_data.x.rows());
        Matrix<T, -1, 1> beta = para.tail(para.size() - 1);
        T intercept = para[0];
        return dot_self(add(multiply(intercept, one), subtract(multiply(extern_data.x, beta), extern_data.y)));
        //return (intercept*one + extern_data.x * beta - extern_data.y).cwiseAbs2().sum();
        /************************************** User defined **************************************/
    }
    // extract compute_para from effective_para
    void get_compute_para(const VectorXd& effective_para, VectorXd& compute_para) const
    {
        if (this->compute_para_index_ptr == NULL) {
            compute_para = effective_para;
        }
        else {
            // assert(effective_para.size() == this->effective_para_index.size());
            VectorXd complete_para = VectorXd::Zero(this->model_size);
            for (int i = 0; i < this->effective_para_index.size(); i++) {
                complete_para[this->effective_para_index[i]] = effective_para[i];
            }
            compute_para = VectorXd(this->compute_para_index_ptr->size());
            for (int i = 0; i < compute_para.size(); i++) {
                compute_para[i] = complete_para[(*this->compute_para_index_ptr)[i]];
            }
        }
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
        VectorXd grad(compute_para.size());
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
Rcpp::XPtr<function_ptr> get_universal_function(MatrixXd x, VectorXd y) {
    extern_data.x = x;
    extern_data.y = y;
    function_ptr tem = (function_ptr)(&function);
    return Rcpp::XPtr<function_ptr>(new function_ptr(tem));
}
"


library(abess)
library(Rcpp)
dataset = generate.data(100, 20, 3)
x = dataset[["x"]]
y = dataset[["y"]]

Rcpp::sourceCpp(code = code)
f = get_universal_function(x,y)
new.x = matrix(0,nrow = nrow(x),ncol = ncol(x)+1)

abess_fit_universal <- abessUniversal(f,ncol(x)+1,nrow(x),new.x,y,always.include = c(1))
abess_fit <- abess(x,y)
