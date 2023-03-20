#include "processor.h"
#include "wrapper.h"
#define MILLISECONDS_TO_SECONDS 0.001

// todo
arma::mat resample(arma::mat input, int sample_rate, int resample_rate) {
    return input; 
}

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
arma::vec get_waveform_and_window_properties(arma::mat& input, int channel, int sample_frequency, int frame_shift, int frame_length, bool round_to_power_of_two = true, double preemphasis_coefficient = 0.97) {
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

arma::mat get_log_energy(const arma::mat& input, double epsilon, double energy_floor) {
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

arma::mat feature_window_function(const std::string& window_type, int window_size, double blackman_coeff) {
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

arma::mat get_window(arma::mat& input, int padded_window_size, int window_size, int window_shift, double energy_floor, const std::string& window_type, double blackman_coeff, bool snip_edges, bool raw_energy, double dither, bool remove_dc_offset, double preemphasis_coefficient) {
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
    input = input % pad(window, input.n_rows - 1, 0);
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

arma::mat inverse_mel_scale(const arma::mat& mel_freq) {
    return 700.0 * (arma::exp(mel_freq / 1127.0) - 1.0);
}

arma::mat mel_scale(const arma::mat& freq) {
    return 1127.0 * arma::log(1.0 + freq / 700.0);
}

double mel_scale_scalar(double freq) {
    return 1127.0 * log(1.0 + freq / 700.0);
}

arma::mat get_mel_banks(int num_bins, int window_length_padded, double sample_freq, double low_freq, double high_freq, double vtln_low, double vtln_high, double vtln_warp_factor) {
    assert(num_bins > 3 && "Must have at least 3 mel bins!");
    assert(window_length_padded % 2 == 0);
    int num_fft_bins = window_length_padded / 2;
    double nyquist = 0.5 * sample_freq;

    if(high_freq <= 0.0) {
        high_freq += nyquist;
    }

    assert ((0.0 <= low_freq && low_freq < nyquist) && (0.0 < high_freq && high_freq <= nyquist) && (low_freq < high_freq) && "Bad values in options: low-freq, high-freq and nyquist!");
    
    double fft_bin_width = sample_freq / (double)window_length_padded;
    double mel_low_freq = mel_scale_scalar(low_freq);
    double mel_high_freq = mel_scale_scalar(high_freq);

    double mel_freq_delta = (mel_high_freq - mel_low_freq) / ((double)num_bins + 1);

    if(vtln_high < 0.0) {
        vtln_high += nyquist;
    }

    assert ((vtln_warp_factor == 1.0 || ((low_freq < vtln_low && vtln_low < high_freq) && (0.0 < vtln_high && vtln_high < high_freq) && (vtln_low < vtln_high))) && "Bad values in options: vtln-low, vtln-high, low-freq and high-freq!");

    arma::mat bin = arange(num_bins);
    arma::mat left_mel = mel_low_freq + bin * mel_freq_delta;
    arma::mat center_mel = mel_low_freq + (bin + 1.0) * mel_freq_delta;
    arma::mat right_mel = mel_low_freq + (bin + 2.0) * mel_freq_delta;

    if(vtln_warp_factor != 1.0) {
        std::cout << "todo!!!" << std::endl;
        exit(0);      
    }

    arma::mat center_freqs = inverse_mel_scale(center_mel);
    arma::mat mel = arma::trans(mel_scale(fft_bin_width * arange(num_fft_bins)));
    
    left_mel = pad(left_mel, mel.n_cols - left_mel.n_cols, 1);
    right_mel = pad(right_mel, mel.n_cols - right_mel.n_cols, 1);
    mel = pad(mel, left_mel.n_rows - mel.n_rows, 0);
    center_mel = pad(center_mel, mel.n_cols - center_mel.n_cols, 1);
    arma::mat up_slope = (mel - left_mel) / (center_mel - left_mel);
    arma::mat down_slope = (right_mel - mel) / (right_mel - center_mel);

    arma::mat bins;
    if(vtln_warp_factor == 1.0) {
        bins = arma::max(arma::mat(up_slope.n_rows, up_slope.n_cols, arma::fill::zeros), arma::min(up_slope, down_slope));
    }
    else {
        std::cout << "todo!!!" << std::endl;
        exit(0);    
    }
    return bins;
}

arma::mat fbank(arma::mat input, int num_mel_bins, int frame_length, int frame_shift, int sample_frequency, double dither, double energy_floor, bool use_power, bool use_log_fbank) {
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

    if(use_power) {
        spectrum = arma::pow(spectrum, 2.0);
    }

    arma::mat mel_energies = get_mel_banks(num_mel_bins, window_paras(2), sample_frequency, 20.0, 0.0, 100.0, -500.0, 1.0);
    mel_energies = arma::join_rows(mel_energies, arma::mat(mel_energies.n_rows, 1, arma::fill::zeros));

    mel_energies = spectrum * arma::trans(mel_energies);

    if(use_log_fbank) {
        mel_energies = arma::log(arma::max(mel_energies, get_epsilon() * arma::mat(mel_energies.n_rows, mel_energies.n_cols, arma::fill::ones)));
    }

    // todo:use_energy
    return mel_energies;
}