#include "wrapper.h"

arma::mat as_strided(const arma::mat& X, int n_rows, int n_cols, int row_stride, int col_stride) {
    arma::mat result(n_rows, n_cols);
    arma::mat X0 = resize(X, 1, X.n_rows * X.n_cols);
    int start = 0;
    int cur = start;
    for(int i = 0; i < n_rows; i++) {
        cur = start;
        for(int j = 0; j < n_cols; j++) {
            result(i, j) = X(0, cur);
            cur += col_stride;
            cur %= X0.n_cols;
        }
        start += row_stride;
        start %= X0.n_cols;
    } 
    return result;
}