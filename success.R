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
using stan::math::dot_self;
using stan::math::multiply;
using stan::math::subtract; // necessary to get the type promotion correct


// TODO: Define a global variable ExternData and no longer pass data pointer
struct ExternData {
    /************************************** User defined **************************************/
    MatrixXd x;
    VectorXd y;
    /************************************** User defined **************************************/
};


class AutoDiffFunction :public UniversalData {
private:
    VectorXd const* effective_para_ptr;
public:
    AutoDiffFunction(const UniversalData& universal_data, VectorXd const* effective_para_ptr)
        :UniversalData(universal_data), effective_para_ptr(effective_para_ptr)
    {
        // assert(this->effective_para.size() == this->effective_para_index.size());
        if (this->compute_para_index_ptr == NULL) {
            this->compute_para_index_ptr = &this->effective_para_index;
        }
    }
    template <typename T>
    T operator()(const Matrix<T, -1, 1>& compute_para) const {
       ExternData*  data = static_cast<ExternData*>(this->data);
      Matrix<T,-1,1> para(this->model_size);
      for (int i = 0; i < para.size(); i++) {
        para[i] = T(0.0);
      }
      for (int i = 0; i < this->effective_para_index.size(); i++) {
        para[this->effective_para_index[i]] = T((*this->effective_para_ptr)[i]);
      }
      for (int i = 0; i < this->compute_para_index_ptr->size(); i++) {
        para[(*this->compute_para_index_ptr)[i]] = compute_para[i];
      }
      return dot_self(subtract(multiply(data->x,para), data->y));
      /************************************** User defined **************************************/
      //return (data->x * para - data->y).cwiseAbs2().sum();
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
Rcpp::XPtr<function_ptr> get_universal_function() {
  function_ptr tem = (function_ptr)(&function); 
  return Rcpp::XPtr<function_ptr>(new function_ptr(tem));
}
"
code2="
// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <RcppEigen.h>
using namespace Eigen;

struct ExternData{
  /************************************** User defined **************************************/
    MatrixXd x;
    VectorXd y;
  /************************************** User defined **************************************/
};

ExternData extern_data;

// [[Rcpp::export]]
Rcpp::XPtr<void*> get_extern_data(MatrixXd x, VectorXd y) {
  extern_data.x = x;
  extern_data.y = y;
  ExternData* ptr = &extern_data;
  return Rcpp::XPtr<void*>(new (void*)(ptr));
}

"
library(abess)
library(Rcpp)
dataset = generate.data(100, 20, 3)
x = dataset[["x"]]
y = dataset[["y"]]

Rcpp::sourceCpp(code = code)
f = get_universal_function()
Rcpp::sourceCpp(code = code2)
d = get_extern_data(x,y)

abess_fit_universal <- abessUniversal(f,d,x,y)
abess_fit <- abess(x,y)
