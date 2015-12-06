#include "constants.h"

class Dataset {
    int N_s, N_i, N_o;
    std::vector< std::vector<double> > features;
    std::vector< std::vector<double> > labels;
public:
    bool load(std::string infile);
    bool save(std::string outfile);

    int getN_i() { return N_i; }
    int getN_o() { return N_o; }
    int getN_s() { return N_s; }
    double getFeature(int s, int i){ return features[s][i]; }
    double getLabel(int s, int i){ return labels[s][i]; }

};

class NeuralNet {
    // count of nodes in each layer (input,hidden,output)
    int N_i, N_h, N_o;
    // store weights in a 3D vector (layers,i,j)
    std::vector< std::vector< std::vector<double> > > weights;
    //std::vector< std::vector< std::vector<double> > > activations;

public:
    bool load(std::string infile);
    bool save(std::string outfile);
    void train(Dataset &data, double learnRate, int numEpochs);
};

