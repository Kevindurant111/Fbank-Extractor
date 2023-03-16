#include "processor.h"
#include "wrapper.h"
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

double get_epsilon() {
    return 1.1921e-07;
}
arma::vec get_waveform_and_window_properties(arma::mat& input, int channel, int sample_frequency, int frame_shift, int frame_length, bool round_to_power_of_two = true, float preemphasis_coefficient = 0.97) {
    channel = std::max(channel, 0);
    assert(channel < static_cast<int>(input.n_rows) && "Invalid channel num!");
    input = input.submat(channel, 0, channel, input.n_cols - 1);
    
    int window_shift = int(sample_frequency * frame_shift * MILLISECONDS_TO_SECONDS);
    int window_size = int(sample_frequency * frame_length * MILLISECONDS_TO_SECONDS);
    int padded_window_size = round_to_power_of_two ? next_power_of_2(window_size) : window_size;
    arma::vec res = {static_cast<double>(window_shift), static_cast<double>(window_size), static_cast<double>(padded_window_size)};
    return res;
}

void get_strided(arma::mat& input, int window_size, int window_shift, bool snip_edges) {
    int num_samples = static_cast<int>(input.n_cols);
    int stride = 1; // tensor.stride(0)
    int m;
    if(snip_edges) {
        assert(num_samples >= window_size && "num_samples can't greater than window_size!");
        m = 1 + floor(((double)num_samples - (double)window_size) / (double)window_shift);
    }
    else {
        std::cout << "todo!!!" << std::endl;
        exit(0);
    }
    input = as_strided(input, m, window_size, window_shift * stride, stride);
}

arma::mat get_log_energy(const arma::mat& input, double epsilon, float energy_floor) {
    arma::mat log_energy = arma::log(arma::max(sum(square(input), 1), arma::mat(input.n_rows, 1, arma::fill::value(epsilon))));
    return log_energy;
}

arma::mat get_window(arma::mat& input, int padded_window_size, int window_size, int window_shift, float energy_floor, std::string window_type, float blackman_coeff, bool snip_edges, bool raw_energy, float dither, bool remove_dc_offset, float preemphasis_coefficient) {
    double epsilon = get_epsilon();
    get_strided(input, window_size, window_shift);
    if(dither != 0.0) {
        std::cout << "todo!!!" << std::endl;
        exit(0);       
    }
    if(remove_dc_offset) {
        arma::mat mean_value = arma::mean(input, 1);
        std::cout << mean_value.n_rows << std::endl;
        std::cout << mean_value.n_cols << std::endl;
        arma::mat mean_value_padding(input.n_rows, input.n_cols);
        for(int i = 0; i < input.n_cols; i++) {
            mean_value_padding.col(i) = mean_value.col(0);
        }
        input = input - mean_value_padding;
        std::cout << input(0, 0) << std::endl;
        std::cout << input(0, 1) << std::endl;
        std::cout << input(0, 2) << std::endl;
        std::cout << input(1, 0) << std::endl;
        std::cout << input(1, 1) << std::endl;
        std::cout << input(2, 2) << std::endl;
        std::cout << input(417, 0) << std::endl;
        std::cout << input(417, 1) << std::endl;
        std::cout << input(417, 2) << std::endl;
    }
    if(raw_energy) {
        auto signal_log_energy = get_log_energy(input, epsilon, energy_floor);
    }
}

arma::mat fbank(arma::mat input, int num_mel_bins, int frame_length, int frame_shift, int sample_frequency, float dither, float energy_floor) {
    auto window_paras = get_waveform_and_window_properties(input, 0, sample_frequency, frame_shift, frame_length);
    auto signal_log_energy = get_window(input, window_paras(2), window_paras(1), window_paras(0), 0.0);
}