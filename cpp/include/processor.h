#include <armadillo>
#include <algorithm>
#include <math.h>
#include <assert.h>

arma::mat resample(arma::mat input, int sample_rate, int resample_rate);

int bit_length(int n);

int next_power_of_2(int x);

arma::vec get_waveform_and_window_properties(arma::mat& input, int channel, int sample_frequency, int frame_shift, int frame_length, bool round_to_power_of_two, float preemphasis_coefficient);

arma::mat fbank(arma::mat input, int num_mel_bins, int frame_length, int frame_shift, float dither, int energy_floor, int sample_frequency);