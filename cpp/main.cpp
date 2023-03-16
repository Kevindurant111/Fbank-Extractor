#include <sndfile.h>
#include <armadillo>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "processor.h"

using namespace arma;

int main(int argc, char** argv) {
    int sample_frequency = 16000;
    int num_mel_bins = 80;
    int frame_shift = 10;
    int frame_length = 25;
    bool round_to_power_of_two = true;
    float preemphasis_coefficient = 0.97;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <wav_file_path>" << std::endl;
        return 1;
    }

    const char* file_path = argv[1];
    SF_INFO sfinfo;
    SNDFILE* sndfile = sf_open(file_path, SFM_READ, &sfinfo);

    if (sndfile == NULL) {
        std::cerr << "Error opening WAV file: " << sf_strerror(sndfile) << std::endl;
        return 1;
    }

    std::cout << "Sample rate: " << sfinfo.samplerate << std::endl;
    std::cout << "Channels: " << sfinfo.channels << std::endl;
    std::cout << "Frames: " << sfinfo.frames << std::endl;

    // Read the samples into a buffer
    const int buffer_size = sfinfo.frames * sfinfo.channels;
    float buffer[buffer_size];
    sf_count_t samples_read = sf_read_float(sndfile, buffer, buffer_size);

    std::cout << "Samples read: " << samples_read << std::endl;

    // Do something with the samples, e.g., print the first 10 samples
    // for (int i = 0; i < 10; i++) {
    //     std::cout << "Sample " << i << ": " << buffer[i] << std::endl;
    // }

    // Close the file
    sf_close(sndfile);

    arma::mat waveform = arma::mat(sfinfo.channels, sfinfo.frames);
    if(sfinfo.channels == 1) {
        for (int i = 0; i < sfinfo.frames; i++) {
            waveform(0, i) = buffer[i] * (1 << 15);
        }
    }
    else if(sfinfo.channels == 2) {
        for (int i = 0; i < sfinfo.frames; i += 2) {
            waveform(0, i / 2) = buffer[i] * (1 << 15);
            waveform(0, i / 2) = buffer[i + 1] * (1 << 15);
        }       
    }
    else {
        std::cerr << "The number of channels in the wav file is not normal!" << std::endl;
        return 1;
    }

    auto feats = fbank(waveform, num_mel_bins, frame_length, frame_shift, sample_frequency, 0.0, 0.0);
    return 0;
}