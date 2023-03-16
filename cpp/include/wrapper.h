#include <armadillo>

arma::mat as_strided(const arma::mat& X,
    int n_rows, 
    int n_cols, 
    int row_stride, 
    int col_stride
);