#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <complex>
#include <sstream>
#include <string>
#include <algorithm>

extern "C" {
#include "radnet.h"
}




uint32_t swap_endian(uint32_t val) {
    return ((val >> 24) & 0xff) |
           ((val << 8) & 0xff0000) |
           ((val >> 8) & 0xff00) |
           ((val << 24) & 0xff000000);
}

void read_idx_images(const std::string& path, std::vector<std::vector<float>>& images) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << path << "\n";
        return;
    }
    uint32_t magic = 0, num_images = 0, rows = 0, cols = 0;
    file.read((char*)&magic, 4);
    file.read((char*)&num_images, 4);
    file.read((char*)&rows, 4);
    file.read((char*)&cols, 4);
    
    magic = swap_endian(magic);
    num_images = swap_endian(num_images);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    images.resize(num_images, std::vector<float>(rows * cols));
    for (uint32_t i = 0; i < num_images; i++) {
        for (uint32_t j = 0; j < rows * cols; j++) {
            unsigned char pixel;
            file.read((char*)&pixel, 1);
            images[i][j] = pixel / 255.0f;
        }
    }
}

void read_idx_labels(const std::string& path, std::vector<int>& labels) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << path << "\n";
        return;
    }
    uint32_t magic = 0, num_items = 0;
    file.read((char*)&magic, 4);
    file.read((char*)&num_items, 4);
    
    magic = swap_endian(magic);
    num_items = swap_endian(num_items);

    labels.resize(num_items);
    for (uint32_t i = 0; i < num_items; i++) {
        unsigned char label;
        file.read((char*)&label, 1);
        labels[i] = label;
    }
}




void read_csv_images(const std::string& path, std::vector<std::vector<float>>& images, std::vector<int>& labels) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Failed to open " << path << "\n";
        return;
    }
    std::string line;
    std::getline(file, line); 

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ',');
        if (cell.empty()) continue;
        labels.push_back(std::stoi(cell));
        
        std::vector<float> img;
        while (std::getline(ss, cell, ',')) {
            img.push_back(std::stof(cell) / 255.0f);
        }
        images.push_back(img);
    }
}




void read_tabular_csv(const std::string& path, std::vector<std::vector<float>>& features, std::vector<int>& labels) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Failed to open " << path << "\n";
        return;
    }
    std::string line;
    std::getline(file, line); 

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stof(cell));
        }
        if (row.size() < 2) continue;
        
        labels.push_back((int)row.back());
        row.pop_back(); 
        
        features.push_back(row);
    }
}

struct HyperParams {
    float lambda;
    float low_epsilon;
    float thresh_shift;
    float thresh_break;
    float learning_rate;
    int epochs;
};

float run_training(Cylinder& cyl, const std::vector<std::vector<float>>& data, const std::vector<int>& labels, bool is_image, HyperParams hp) {
    const float silk_gate_bias = 0.05f;
    unsigned int rng_state = 42u;

    int out_classes = cyl.tradeoff.output_size;
    std::vector<float> out_probs(out_classes);
    
    float final_accuracy = 0.0f;

    int num_features = data.empty() ? 0 : data[0].size();
    std::vector<std::vector<std::vector<float>>> proj_matrix;
    if (!is_image && num_features > 0) {
        proj_matrix.resize(cyl.num_webs);
        for (int w = 0; w < cyl.num_webs; w++) {
            int nodes = cyl.webs[w].shells[0].num_nodes;
            proj_matrix[w].resize(nodes);
            for (int k = 0; k < nodes; k++) {
                proj_matrix[w][k].resize(num_features);
                for (int f = 0; f < num_features; f++) {
                    proj_matrix[w][k][f] = -1.0f + 2.0f * ((float)rand() / (float)RAND_MAX);
                }
            }
        }
    }

    for (int epoch = 0; epoch < hp.epochs; epoch++) {
        int correct = 0;
        int total = 0;
        for (size_t i = 0; i < data.size(); i++) {
            for (int w = 0; w < cyl.num_webs; w++) {
                Web* web = &cyl.webs[w];
                if (is_image) {
                    radnet_seed_from_image(data[i].data(), 28, 28, web);
                } else {
                    Shell* shell1 = &web->shells[0];
                    for (int k = 0; k < shell1->num_nodes; k++) {
                        float sum = 0.0f;
                        for (int f = 0; f < num_features; f++) {
                            sum += data[i][f] * proj_matrix[w][k][f];
                        }
                        shell1->seed[k] = std::complex<float>(sum, 0.0f);
                        shell1->z_state[k] = std::complex<float>(sum, 0.0f);
                    }
                }

                for (int s = 0; s < web->num_shells - 1; s++) {
                    radnet_forward_shell(web, s, silk_gate_bias);
                }
            }

            cylinder_forward(&cyl, out_probs.data());
            


            int pred = -1;
            float max_prob = -1.0f;
            for (int j = 0; j < out_classes; j++) { 
                if (out_probs[j] > max_prob) {
                    max_prob = out_probs[j];
                    pred = j;
                }
            }

            if (pred == labels[i]) {
                correct++;
            }
            total++;

            cylinder_echo(&cyl, out_probs.data(), labels[i], hp.lambda);
            cylinder_mutate(&cyl, hp.thresh_shift, hp.thresh_break, hp.low_epsilon, hp.learning_rate, &rng_state);

            if (data.size() >= 10000 && (i + 1) % 10000 == 0) {
                std::cout << "  Sample " << (i + 1) << " | Accuracy: " 
                          << (100.0f * correct / total) << "% (" << correct << "/" << total << ")" 
                          << " | W[0]: " << std::abs(cyl.tradeoff.weights[0]) << std::endl;
            }
        }
        final_accuracy = 100.0f * correct / total;
        std::cout << "  Epoch " << (epoch + 1) << "/" << hp.epochs << " | Accuracy: " << final_accuracy << "%" 
                  << " | W[0]: " << std::abs(cyl.tradeoff.weights[0]) << std::endl;
    }
    return final_accuracy;
}




int main(int argc, char** argv) {
    std::vector<std::vector<float>> data;
    std::vector<int> labels;
    bool is_image = true;
    int num_shells = 16;
    int num_classes = 10;
    std::string mode = "";

    if (argc == 1) {
        std::cout << "========================================\n";
        std::cout << "       RADNET INTERACTIVE TRAINER       \n";
        std::cout << "========================================\n\n";
        
        std::cout << "What dataset would you like to train on? (e.g., Heart_Clean.csv): ";
        std::string filename;
        std::cin >> filename;
        
        std::cout << "How many classes does this dataset have? (e.g., 2, 3, 10): ";
        std::cin >> num_classes;
        
        int epochs;
        std::cout << "How many epochs would you like to train for? (e.g., 50): ";
        std::cin >> epochs;
        
        std::cout << "\nAwesome! Connecting to " << filename << " and preparing the RadNet topology...\n";
        
        read_tabular_csv(filename, data, labels);
        is_image = false;
        num_shells = 2; 
        
        if (data.empty() || labels.empty()) {
            std::cerr << "Failed to load dataset.\n";
            return 1;
        }
        std::cout << "Loaded " << data.size() << " samples.\n";
        
        unsigned int rng_state = 42u;
        Cylinder cyl;
        if (cylinder_init(&cyl, 4, num_shells, num_classes, &rng_state) != 0) {
            std::cerr << "Failed to init Cylinder.\n";
            return 1;
        }

        HyperParams hp = {0.05f, 0.01f, 0.3f, 1.2f, 0.5f, epochs};
        std::cout << "\nStarting " << num_shells << "-shell RadNet forward/backward loop for " << epochs << " epochs...\n";
        run_training(cyl, data, labels, is_image, hp);
        cylinder_free(&cyl);
        
        std::cout << "\nTraining Complete! You did it. Press Enter to exit...\n";
        std::cin.ignore();
        std::cin.get();
        return 0;
    }

    if (argc < 2) {
        std::cerr << "Usage:\n";
        std::cerr << "  " << argv[0] << " --idx <images> <labels>\n";
        std::cerr << "  " << argv[0] << " --csv-image <data.csv>\n";
        std::cerr << "  " << argv[0] << " --tabular <data.csv> <classes>\n";
        std::cerr << "  " << argv[0] << " --tune-mnist <images> <labels>\n";
        return 1;
    }

    mode = argv[1];

    std::cout << "Loading dataset...\n";
    if (mode == "--idx" && argc == 4) {
        read_idx_images(argv[2], data);
        read_idx_labels(argv[3], labels);
    } else if (mode == "--csv-image" && argc == 3) {
        read_csv_images(argv[2], data, labels);
        
        num_classes = 25; 
    } else if (mode == "--tabular" && argc == 4) {
        read_tabular_csv(argv[2], data, labels);
        is_image = false;
        num_shells = 2; 
        num_classes = std::stoi(argv[3]);
    } else if (mode == "--tune-mnist" && argc == 4) {
        read_idx_images(argv[2], data);
        read_idx_labels(argv[3], labels);
    } else {
        std::cerr << "Invalid arguments.\n";
        return 1;
    }

    if (data.empty() || labels.empty()) {
        std::cerr << "Failed to load dataset.\n";
        return 1;
    }
    std::cout << "Loaded " << data.size() << " samples.\n";

    unsigned int rng_state = 42u;
    if (mode == "--tune-mnist") {
        
        if (data.size() > 1000) {
            data.resize(1000);
            labels.resize(1000);
        }
        std::vector<float> lambdas = {0.05f, 0.15f, 0.5f};
        std::vector<float> epsilons = {0.001f, 0.01f, 0.05f};
        std::vector<float> shifts = {0.3f, 0.6f};
        
        float best_acc = -1.0f;
        HyperParams best_hp;
        
        std::cout << "Starting Hyperparameter Grid Search on 1000 samples (2 epochs each)...\n";
        
        for (float l : lambdas) {
            for (float e : epsilons) {
                for (float s : shifts) {
                    Cylinder cyl;
                    cylinder_init(&cyl, 1, num_shells, num_classes, &rng_state); 
                    
                    HyperParams hp = {l, e, s, 1.2f, 0.01f, 2};
                    std::cout << "Testing: lambda=" << l << ", epsilon=" << e << ", shift=" << s << std::endl;
                    float acc = run_training(cyl, data, labels, is_image, hp);
                    
                    if (acc > best_acc) {
                        best_acc = acc;
                        best_hp = hp;
                    }
                    cylinder_free(&cyl);
                }
            }
        }
        
        std::cout << "Epsilon: " << best_hp.low_epsilon << "\n";
        std::cout << "Shift: " << best_hp.thresh_shift << "\n";
        std::cout << "Validation Accuracy: " << best_acc << "%\n";
        std::cout << "============================\n";
        
    } else {
        Cylinder cyl;
        if (cylinder_init(&cyl, 4, num_shells, num_classes, &rng_state) != 0) {
            std::cerr << "Failed to init Cylinder.\n";
            return 1;
        }

        HyperParams hp = {0.05f, 0.01f, 0.3f, 1.2f, 0.5f, 50}; 
        std::cout << "Starting " << num_shells << "-shell RadNet forward/backward loop for 50 epochs...\n";
        run_training(cyl, data, labels, is_image, hp);
        cylinder_free(&cyl);
    }
    return 0;
}

