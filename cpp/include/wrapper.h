#include <armadillo>

arma::mat as_strided(const arma::mat& X,
    int n_rows, 
    int n_cols, 
    int row_stride, 
    int col_stride
);

// dim = 0, expand the number of rows; dim = 1, expand the number of columns
arma::mat pad(const arma::mat& X,
    int num,
    int dim = 0
);

// Generate a matrix of shape [size, 1] with elements increasing linearly from 0 to size-1 
arma::mat arange(int size);