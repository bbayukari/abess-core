code = "
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

using namespace Eigen;
using namespace stan::math; // necessary to get the type promotion correct

typedef double(*UniversalFunction)(const Eigen::VectorXd& effective_para, Eigen::VectorXd* gradient, Eigen::MatrixXd* hessian,
    const int model_size, const Eigen::VectorXi &effective_para_index, const Eigen::VectorXi* compute_para_index_ptr);

struct ExternData {
    /************************************* User defined **************************************/
    MatrixXd x;
    VectorXd y;
    /************************************** User defined **************************************/
} extern_data;

class AutoDiffFunction{
private:
    const VectorXd* effective_para_ptr;
    const int model_size;
    const VectorXi* effective_para_index_ptr;
    const VectorXi* compute_para_index_ptr; //when it's NULL, compute_para equals to effective_para
public:
    AutoDiffFunction(const VectorXd* effective_para_ptr, const int model_size, const VectorXi* effective_para_index_ptr, const VectorXi* compute_para_index_ptr)
        :effective_para_ptr(effective_para_ptr), model_size(model_size), effective_para_index_ptr(effective_para_index_ptr), compute_para_index_ptr(compute_para_index_ptr)
    {
        if (this->compute_para_index_ptr == NULL) {
            this->compute_para_index_ptr = this->effective_para_index_ptr;
        }
    }
    template <typename T>
    T operator()(const Matrix<T, -1, 1>& compute_para) const {
        Matrix<T, -1, 1> para(this->model_size);
        for (int i = 0; i < para.size(); i++) {
            para[i] = T(0.0);
        }
        for (int i = 0; i < this->effective_para_index_ptr->size(); i++) {
            para[(*this->effective_para_index_ptr)[i]] = T((*this->effective_para_ptr)[i]);
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
    void get_compute_para(VectorXd& compute_para)
    {
        if (this->compute_para_index_ptr == NULL) {
            compute_para = *this->effective_para_ptr;
        }
        else {
            VectorXd complete_para = VectorXd::Zero(this->model_size);
            for (int i = 0; i < this->effective_para_index_ptr->size(); i++) {
                complete_para[(*this->effective_para_index_ptr)[i]] = (*this->effective_para_ptr)[i];
            }
            compute_para.resize(this->compute_para_index_ptr->size());
            for (int i = 0; i < compute_para.size(); i++) {
                compute_para[i] = complete_para[(*this->compute_para_index_ptr)[i]];
            }
        }
    }
};

double universal_function(const Eigen::VectorXd& effective_para, Eigen::VectorXd* gradient, Eigen::MatrixXd* hessian,
    const int model_size, const Eigen::VectorXi &effective_para_index, const Eigen::VectorXi* compute_para_index_ptr)
{
    AutoDiffFunction function(&effective_para, model_size, &effective_para_index, compute_para_index_ptr);
    VectorXd compute_para;
    function.get_compute_para(compute_para);
    double value = 0.;
    if (hessian) {
        if (!gradient) {
            VectorXd grad(compute_para.size());
            gradient = &grad;
        }
        stan::math::hessian(function, compute_para, value, *gradient, *hessian);
        return value;
    }
    if (gradient) {
        stan::math::gradient(function, compute_para, value, *gradient);
        return value;
    }
    return function(compute_para);
}

// [[Rcpp::export]]
Rcpp::XPtr<UniversalFunction> get_universal_function() {
    return Rcpp::XPtr<UniversalFunction>(new UniversalFunction(&universal_function));
}

// [[Rcpp::export]]
void set_extern_data(MatrixXd x, VectorXd y){
    extern_data.x = x;
    extern_data.y = y;
}
"


library(abess)
library(Rcpp)
dataset = generate.data(100, 20, 3)
x = dataset[["x"]]
y = dataset[["y"]]

Rcpp::sourceCpp(code = code)
set_extern_data(x,y)
f = get_universal_function()

new.x = matrix(0,nrow = nrow(x),ncol = ncol(x)+1)

abess_fit_universal <- abessUniversal(f,ncol(x)+1,nrow(x),new.x,y,always.include = c(1))
abess_fit <- abess(x,y)
