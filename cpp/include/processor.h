#include <armadillo>
#include <algorithm>
#include <math.h>
#include <string>
#include <assert.h>

arma::mat resample(arma::mat input, 
    int sample_rate, 
    int resample_rate
);

int bit_length(int n);

int next_power_of_2(int x);

double get_epsilon(); 

arma::vec get_waveform_and_window_properties(arma::mat& input, 
    int channel, 
    int sample_frequency, 
    int frame_shift, 
    int frame_length, 
    bool round_to_power_of_two, 
    double preemphasis_coefficient
);

void get_strided(arma::mat& input, 
    int window_size, 
    int window_shift, 
    bool snip_edges = true
);

arma::mat get_log_energy(const arma::mat& input, 
    double epsilon, 
    double energy_floor
);

arma::mat feature_window_function(const std::string& window_type,
    int window_size,
    double blackman_coeff
);

arma::mat hann_window(int window_size);

arma::mat get_window(arma::mat& input, 
    int padded_window_size, 
    int window_size, 
    int window_shift,
    double energy_floor = 1.0, 
    const std::string& window_type = "povey",
    double blackman_coeff = 0.42,
    bool snip_edges = true,
    bool raw_energy = true, 
    double dither = 0.0,
    bool remove_dc_offset = true,
    double preemphasis_coefficient = 0.97
);

arma::mat inverse_mel_scale(const arma::mat& mel_freq);

arma::mat mel_scale(const arma::mat& freq);

double mel_scale_scalar(double freq);

arma::mat get_mel_banks(int num_bins,
    int window_length_padded,
    double sample_freq,
    double low_freq,
    double high_freq,
    double vtln_low,
    double vtln_high,
    double vtln_warp_factor
);

arma::mat fbank(arma::mat input,
    int num_mel_bins, 
    int frame_length, 
    int frame_shift, 
    int sample_frequency, 
    double dither = 0.0, 
    double energy_floor = 1.0,
    bool use_power = true,
    bool use_log_fbank = true
);