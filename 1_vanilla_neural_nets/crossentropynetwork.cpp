#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <algorithm>
#include <vector>
#include <cmath>

#include <Eigen/Dense>

// g++ -Ofast -std=c++11 -I /opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3 network.cpp -o network && ./network

using namespace std;

// Reads in a CSV of training data corresponding to the MNIST dataset
// Each line is in the form of <Label>,<Pixel 1>,<Pixel 2>,...
// Returns a vector of pairs, where the first element of the pair is the label
// and the second element is a vector of the pixel values
const vector< pair<int, vector<float> > > getData(const string filename) {
    ifstream input(filename);

    if(!input.is_open()) {
        throw runtime_error("Could not open file");
    }
    
    // Read the column names
    if(!input.good()) {
        throw runtime_error("Could not read file");
    }

    string line;
    vector< pair<int, vector<float> > > data;
    
    // Extract the first line in the file
    while(getline(input, line)) {
        std::stringstream ss(line);

        string label, csv_val;
        getline(ss, label,  ',');
        vector<float> image;
        while(getline(ss, csv_val,  ',')) {
            image.push_back( stof(csv_val));
        }
        data.push_back(make_pair(stoi(label), image));
    }

    return data;
}

float sigmoid(float val) {
    return 1.0/(1.0+exp(-val));
}

float sigmoid_prime(float val) {
    return sigmoid(val)*(1-sigmoid(val));
}

int main() {
    cout << "Loading dataset" << endl;
    auto data = getData("train.csv");
    auto test = getData("test.csv");

    vector<int> layer_sizes{784, 200, 200, 50, 10};
    auto num_layers = layer_sizes.size();

    // Each layer has its own vector of biases
    // Each bias corresponds to a neuron in the layer
    vector<Eigen::VectorXf> biases;
    for (auto i = layer_sizes.begin() + 1; i != layer_sizes.end(); ++i) {
        Eigen::VectorXf layer_biases(*i);
        layer_biases.setRandom();
        biases.push_back(layer_biases);
    }

    // Each layer has its own matrix of weights
    // That matrix maps this layer's neurons to each of the previous layer's neurons
    vector< Eigen::MatrixXf > weights;
    for (auto i = layer_sizes.begin() + 1; i != layer_sizes.end(); ++i) {
        Eigen::MatrixXf layer_weights(*i, *(i - 1));
        layer_weights.setRandom();
        weights.push_back(layer_weights);
    }

    int batch_size = 30;
    int training_repeat_count = 100;
    auto rng = default_random_engine();

    for (auto epoch = 0; epoch < training_repeat_count; ++epoch) {
        cout << "Epoch" << epoch << endl;
        // Shuffle the data
        shuffle(data.begin(), data.end(), rng);

        // Split the data into batches
        vector< vector< pair<int, vector<float> > > > batches;
        for (auto j = data.begin(); j != data.end(); ) {
            auto from = j;
            auto to = j + batch_size;
            auto distance_to_end = distance(j, data.end());
            if (distance_to_end <= batch_size) {
                to = data.end();
            } 
            vector< pair<int, vector<float> > > batch(from, to);
            batches.push_back(batch);

            j = to;
        }

        // Train the network on each batch
        for (auto batch = batches.begin(); batch != batches.end(); ++batch) {
            auto batch_bias_adjustments = biases;
            for (auto i = batch_bias_adjustments.begin(); i != batch_bias_adjustments.end(); ++i) {
                i->setZero();
            }
            auto batch_weight_adjustments = weights;
            for (auto i = batch_weight_adjustments.begin(); i != batch_weight_adjustments.end(); ++i) {
                i->setZero();
            }

            for (auto item = batch->begin(); item != batch->end(); ++item) {                
                auto label = item->first;
                auto image = item->second;

                auto label_vector = Eigen::VectorXf(10);
                label_vector.setZero();
                label_vector[label] = 1;

                auto instance_bias_adjustments = batch_bias_adjustments;
                auto instance_weight_adjustments = batch_weight_adjustments;    

                // Feedforward
                vector<Eigen::VectorXf> activations, biased_weighted_sums;
                Eigen::VectorXf imageVector(image.size());
                for (auto i = 0; i < image.size(); ++i) {
                    imageVector[i] = image[i];
                }
                activations.push_back(imageVector);
                
                for (auto i = 0; i < biases.size(); ++i) {
                    auto weight_sum_of_previous_neurons = weights[i] * activations.back();
                    auto bias_weighted_sum_of_previous_neurons = weight_sum_of_previous_neurons + biases[i];
                    biased_weighted_sums.push_back(bias_weighted_sum_of_previous_neurons);

                    auto layer = bias_weighted_sum_of_previous_neurons.unaryExpr(&sigmoid);
                    activations.push_back(layer);
                }

                // Backpropagation

                // Derivative of the cost function w/r/t the last layer includes the last layer biases and weights,
                auto biased_weighted_sum = biased_weighted_sums.back();
                auto prime = biased_weighted_sum.unaryExpr(&sigmoid_prime);

                Eigen::VectorXf dc_db = (biased_weighted_sum.unaryExpr(&sigmoid) - label_vector);
                auto dc_dw = dc_db * activations[num_layers - 2].transpose();

                instance_bias_adjustments[instance_bias_adjustments.size()-1] = dc_db;
                instance_weight_adjustments[instance_weight_adjustments.size()-1] = dc_dw;

                for (auto i = 2; i < num_layers; ++i) {
                    auto reverse_i  = num_layers - i;
                    auto biased_weighted_sum = biased_weighted_sums[reverse_i-1];
                    auto prime = biased_weighted_sum.unaryExpr(&sigmoid_prime);
                    
                    dc_db = (weights[reverse_i ].transpose() * dc_db).cwiseProduct(prime);
                    instance_bias_adjustments[reverse_i-1] = dc_db;
                    instance_weight_adjustments[reverse_i-1] = dc_db * activations[reverse_i-1].transpose();
                }

                
                for (auto i = 0; i < batch_bias_adjustments.size(); ++i) {
                    // cout << "hi" << endl;
                    batch_bias_adjustments[i] += instance_bias_adjustments[i];
                    // cout << "bye" << endl;
                    batch_weight_adjustments[i] += instance_weight_adjustments[i];
                    // cout << "bye again" << endl;
                }
                // cout << "here" << endl;
            }


            auto step_size = (epoch < 18 ? 1.5 : .1) / batch_size;
            for (auto i = 0; i < biases.size(); ++i) {
                // cout << "here 2" << endl;
                biases[i] = biases[i] - (step_size * batch_bias_adjustments[i]);
                // cout << "here 3" << endl;
                weights[i] = weights[i] -  (step_size * batch_weight_adjustments[i]);
            }
        }


        auto correct_count = 0;
        auto total = 0;
        for (auto test_item = test.begin(); test_item != test.end(); ++test_item) {
            total ++;
            auto label = test_item->first;
            auto label_vector = Eigen::VectorXf(10);
            label_vector.setZero();
            label_vector[label] = 1;

            auto image = test_item->second;

            // Feedforward
            vector<Eigen::VectorXf> activations;                
            Eigen::VectorXf imageVector(image.size());
            for (auto i = 0; i < image.size(); ++i) {
                imageVector[i] = image[i];
            }
            activations.push_back(imageVector);
            
            for (auto i = 0; i < biases.size(); ++i) {
                auto weight_sum_of_previous_neurons = weights[i] * activations.back();
                auto bias_weighted_sum_of_previous_neurons = weight_sum_of_previous_neurons + biases[i];
                auto layer = bias_weighted_sum_of_previous_neurons.unaryExpr(&sigmoid);
                activations.push_back(layer);
            }

            auto output = activations.back();
            Eigen::Index maxIdx;
            auto max = output.maxCoeff(&maxIdx);
            //cout << output << endl;

            if (maxIdx == label) {
                correct_count ++;
            }             
        }
        cout
                << " Correct: " << correct_count 
                << " Total: " << total
                << " Percent: " << (correct_count * 1.0) / (total * 1.0) << endl;
    }

    return 0;
}
