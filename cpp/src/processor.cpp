#include "processor.h"
#define MILLISECONDS_TO_SECONDS 0.001

arma::mat resample(arma::mat input, int sample_rate, int resample_rate) {}

int bit_length(int n) {
    // Find the position of the most significant bit set
    int position = 0;
    while (n) {
        position++;
        n >>= 1;
    }

    return position;
}

int next_power_of_2(int x) {
    return (x == 0 ? 1 : std::pow(2.0, (double)bit_length(x - 1)));
}

arma::vec get_waveform_and_window_properties(arma::mat& input, int channel, int sample_frequency, int frame_shift, int frame_length, bool round_to_power_of_two, float preemphasis_coefficient) {
    channel = std::max(channel, 0);
    assert(channel < static_cast<int>(input.n_rows) && "Invalid channel num!");
    input = input.submat(channel, 0, channel, input.n_cols - 1);
    
    int window_shift = int(sample_frequency * frame_shift * MILLISECONDS_TO_SECONDS);
    int window_size = int(sample_frequency * frame_length * MILLISECONDS_TO_SECONDS);
    int padded_window_size = round_to_power_of_two ? next_power_of_2(window_size) : window_size;
    arma::vec res = {static_cast<double>(window_shift), static_cast<double>(window_size), static_cast<double>(padded_window_size)};
    return res;
}

arma::mat fbank(arma::mat input, int num_mel_bins, int frame_length, int frame_shift, float dither, int energy_floor, int sample_frequency) {

}