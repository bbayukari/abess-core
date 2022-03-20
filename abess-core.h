#pragma once

#include <iostream>
#include <fstream>
#include "api.h"

int read(const char* filename, Eigen::MatrixXd& data) {
    int i = 0, j = 0;
    ifstream fp(filename);
    string line;
    getline(fp, line);
    while (getline(fp, line)) {
        string number;
        istringstream readstr(line);
        while (getline(readstr, number, ',')) {
            data(i, j) = atof(number.c_str());
            j++;
        }
        i++;
        j = 0;
    }
    return 0;
}

List glm() {
    Eigen::MatrixXd x(100, 20);
    Eigen::MatrixXd y(100, 1);
    read("x.csv", x);
    read("y.csv", y);
    //cout << x.transpose()*y;

    int n = 100;
    int p = 20;
    int normalize_type = 1;
    Eigen::VectorXd weight = Eigen::VectorXd::Constant(100, 1);
    int algorithm_type = 6;
    int model_type = 1;
    int max_iter = 20;
    int exchange_num = 2;
    int path_type = 1;
    bool is_warm_start = true;
    int ic_type = 3;
    double ic_coef = 1;
    int Kfold = 1;
    Eigen::VectorXi sequence = Eigen::VectorXi::LinSpaced(21, 0, 20);
    Eigen::VectorXd lambda_seq = Eigen::VectorXd::Constant(1, 0);
    int s_min = 1;
    int s_max = 20;
    double lambda_min = 0;
    double lambda_max = 0;
    int nlambda = 10;
    int screening_size = -1;
    Eigen::VectorXi g_index = Eigen::VectorXi::LinSpaced(20, 0, 19);
    Eigen::VectorXi always_select;
    int primary_model_fit_max_iter = 10;
    double primary_model_fit_epsilon = 1e-06;
    bool early_stop = false;
    bool approximate_Newton = false;
    int thread = 0;
    bool covariance_update = false;
    bool sparse_matrix = false;
    int splicing_type = 0;
    int sub_search = 20;
    Eigen::VectorXi cv_fold_id;
    Eigen::VectorXi A_init;

    return abessGLM_API(x, y, n, p, normalize_type, weight,
        algorithm_type, model_type, max_iter, exchange_num, path_type, is_warm_start,
        ic_type, ic_coef, Kfold, sequence, lambda_seq,
        s_min, s_max, lambda_min, lambda_max, nlambda, screening_size,
        g_index, always_select, primary_model_fit_max_iter,
        primary_model_fit_epsilon, early_stop, approximate_Newton, thread,
        covariance_update, sparse_matrix, splicing_type, sub_search,
        cv_fold_id, A_init);
}