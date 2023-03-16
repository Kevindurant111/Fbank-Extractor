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
    float preemphasis_coefficient
);

void get_strided(arma::mat& input, 
    int window_size, 
    int window_shift, 
    bool snip_edges = true
);

arma::mat get_log_energy(const arma::mat& input, 
    double epsilon, 
    float energy_floor
);

arma::mat get_window(arma::mat& input, 
    int padded_window_size, 
    int window_size, 
    int window_shift,
    float energy_floor = 1.0, 
    std::string window_type = "povey",
    float blackman_coeff = 0.42,
    bool snip_edges = true,
    bool raw_energy = true, 
    float dither = 0.0,
    bool remove_dc_offset = true,
    float preemphasis_coefficient = 0.97
);

arma::mat fbank(arma::mat input,
    int num_mel_bins, 
    int frame_length, 
    int frame_shift, 
    int sample_frequency, 
    float dither = 0.0, 
    float energy_floor = 1.0
);