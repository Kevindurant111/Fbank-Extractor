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
    // arma::mat log_energy = arma::log(arma::max(sum(square(input), 1), arma::mat(input.n_rows, 1, arma::fill::value(epsilon))));
    // if (energy_floor == 0.0) {
    //     return log_energy;
    // }
    // return arma::max(log_energy, arma::mat(log_energy.n_rows, log_energy.n_cols, arma::fill::value(log(epsilon))));
    arma::mat log_energy = arma::log(arma::max(sum(square(input), 1), epsilon * arma::mat(input.n_rows, 1, arma::fill::ones)));
    if (energy_floor == 0.0) {
        return log_energy;
    }
    return arma::max(log_energy, log(epsilon) * arma::mat(log_energy.n_rows, log_energy.n_cols, arma::fill::ones));
}

arma::mat feature_window_function(const std::string& window_type, int window_size, float blackman_coeff) {
    if(window_type == "povey") {
        return hann_window(window_size);
    }
    else {
        std::cerr << "Invalid window type " + window_type<< std::endl;
        exit(1);
    }
    return arma::mat();
}

arma::mat hann_window(int window_size) {
    // generate symmetric hann window
    assert(window_size > 0 && "window size must greater than 0!");
    arma::mat window(1, window_size, arma::fill::zeros);
    for(int i = 0; i < window_size; i++) {
        window(0, i) = pow(0.5 * (1 - cos(2 * M_PI * i / (window_size - 1))), 0.85);
    }
    return window;
}

arma::mat get_window(arma::mat& input, int padded_window_size, int window_size, int window_shift, float energy_floor, const std::string& window_type, float blackman_coeff, bool snip_edges, bool raw_energy, float dither, bool remove_dc_offset, float preemphasis_coefficient) {
    arma::mat signal_log_energy;
    double epsilon = get_epsilon();
    get_strided(input, window_size, window_shift);
    if(dither != 0.0) {
        std::cout << "todo!!!" << std::endl;
        exit(0);       
    }
    if(remove_dc_offset) {
        arma::mat mean_value = arma::mean(input, 1);
        arma::mat mean_value_padding(input.n_rows, input.n_cols);
        for(arma::uword i = 0; i < input.n_cols; i++) {
            mean_value_padding.col(i) = mean_value.col(0);
        }
        input = input - mean_value_padding;
    }
    if(raw_energy) {
        signal_log_energy = get_log_energy(input, epsilon, energy_floor);
    }
    if(preemphasis_coefficient != 0.0) {
        arma::mat first_col = input.col(0);
        arma::mat offset_strided_input = arma::join_rows(first_col, input);
        input = input - preemphasis_coefficient * offset_strided_input.submat(0, 0, offset_strided_input.n_rows - 1, offset_strided_input.n_cols - 2);
    }
    auto window = feature_window_function(window_type, window_size, 0.0);
    input = input % pad(window, input.n_rows, 0);
    if(padded_window_size != window_size) {
        int padding_right = padded_window_size - window_size;
        arma::mat pad_matrix(input.n_rows, padding_right, arma::fill::zeros);
        input = arma::join_rows(input, pad_matrix);
    }
    // Compute energy after window function (not the raw one)
    if(!raw_energy) {
        signal_log_energy = get_log_energy(input, epsilon, energy_floor);
    }

    return signal_log_energy;
}

arma::mat fbank(arma::mat input, int num_mel_bins, int frame_length, int frame_shift, int sample_frequency, float dither, float energy_floor) {
    auto window_paras = get_waveform_and_window_properties(input, 0, sample_frequency, frame_shift, frame_length);
    auto signal_log_energy = get_window(input, window_paras(2), window_paras(1), window_paras(0), 0.0);
    
    // Real Fast Fourier Transform
    arma::mat spectrum(input.n_rows, floor(input.n_cols / 2) + 1, arma::fill::zeros);
    for(arma::uword i = 0; i < spectrum.n_rows; i++) {
        arma::rowvec v = input.row(i);
        arma::cx_rowvec fft_v = arma::fft(v);
        arma::rowvec real_v = arma::pow(arma::pow(arma::real(fft_v), 2.0) + arma::pow(arma::imag(fft_v), 2.0), 0.5);
        real_v = real_v.subvec(0, floor(real_v.n_cols / 2));
        spectrum.row(i) = real_v;
    }
    
    // arma::mat spectrum(input.n_rows, floor(input.n_cols / 2) + 1);
    // for(arma::uword i = 0; i < 1; i++) {
    //     arma::colvec v = input.col(i);
    //     arma::rowvec v1 = v;
    //     arma::colvec real_v = arma::real(arma::fft(v1));
    //     real_v.print();
    //     // real_v = real_v.subvec(0, floor(real_v.n_cols / 2));
    //     spectrum.col(i) = real_v;
    // }
    // spectrum = arma::abs(spectrum);

    std::cout << spectrum.n_rows << std::endl;
    std::cout << spectrum.n_cols << std::endl;
    std::cout << spectrum(0, 0) << std::endl;
    std::cout << spectrum(0, 1) << std::endl;
    std::cout << spectrum(0, 2) << std::endl;
    std::cout << spectrum(417, 0) << std::endl;
    std::cout << spectrum(417, 1) << std::endl;
    std::cout << spectrum(417, 2) << std::endl;
    std::cout << spectrum(417, 254) << std::endl;
    std::cout << spectrum(417, 255) << std::endl;
    std::cout << spectrum(417, 256) << std::endl;
}