#include "nn.h"

// define logistic sigmoid and its derivative
#define SIG(x) 1/(1+exp(-x))
#define SIGD(x) SIG(x)*(1-SIG(x))
#define A 0
#define B 1
#define C 2
#define D 3

using namespace std;

void NeuralNet::generate(int input, int hidden, int output){
    srand(time(NULL));

    N_i = input;
    N_o = output;
    N_h = hidden;

    //resize weights array accounting for bias
    weights.resize(2);
    weights[0].resize(N_i+1);
    for(int i = 0; i < weights[0].size(); i++){
        weights[0][i].resize(N_h);
    }
    weights[1].resize(N_h+1);
    for(int h = 0; h < weights[1].size(); h++){
        weights[1][h].resize(N_o);
    }

    // load weights from file
    for (int h = 0; h < N_h; h++){
        for (int i = 0;i < N_i+1; i++){
            weights[0][i][h] = (double)(rand() % 500)/1000;
        }
    }
    for (int o = 0; o < N_o; o++){
        for (int h = 0; h < N_h+1; h++){
            weights[1][h][o] = (double)(rand() % 500)/1000;
        }
    }
}

bool NeuralNet::load(string infile){
    ifstream filein;
    filein.open(infile);
    string instring;

    // read in network parameters
    filein >> instring;
    N_i = stoi(instring);
    filein >> instring;
    N_h = stoi(instring);
    filein >> instring;
    N_o = stoi(instring);

    //resize weights array accounting for bias
    weights.resize(2);
    weights[0].resize(N_i+1);
    for(int i = 0; i < weights[0].size(); i++){
        weights[0][i].resize(N_h);
    }
    weights[1].resize(N_h+1);
    for(int h = 0; h < weights[1].size(); h++){
        weights[1][h].resize(N_o);
    }

    // load weights from file
    for (int h = 0; h < N_h; h++){
        for (int i = 0;i < N_i+1; i++){
            filein >> instring;
            weights[0][i][h] = stod(instring);
        }
    }
    for (int o = 0; o < N_o; o++){
        for (int h = 0; h < N_h+1; h++){
            filein >> instring;
            weights[1][h][o] = stod(instring);
        }
    }

    return true;
}

bool NeuralNet::save(string outfile){
    ofstream fileout;
    fileout.open(outfile);

    fileout.setf(ios::fixed,ios::floatfield);
    fileout.precision(3);

    fileout << N_i << ' ' << N_h << ' ' << N_o << endl;

    for(int h = 0; h < N_h; h++){
        for (int i = 0; i < N_i+1; i++){
            fileout << weights[0][i][h];
            if (i < N_i)
                fileout << ' ';
        }
        fileout << endl;
    }

    for(int o = 0; o < N_o; o++){
        for(int h = 0; h < N_h+1; h++){
            fileout << weights[1][h][o];
            if (h < N_h)
                fileout << ' ';
        }
        fileout << endl;
    }

    return true;
}

void NeuralNet::train(Dataset &data, double learnRate, int numEpochs){
    // weights are assumed to have been initalized when the NeuralNet is loaded

    if (N_i != data.getN_i() || N_o != data.getN_o()){
        cerr << "ERROR: Dataset and Network size mismatch\n";
        exit(1);
    }

    // initialize a few variables to help with bookkeeping
    vector< vector<double> > activations(3,vector<double>(1,0));
    activations[0].resize(N_i);
    activations[1].resize(N_h);
    activations[2].resize(N_o);

    vector< vector<double> > layerSum(activations);
    vector< vector<double> > deltas(activations);
    double tmp;


    int numSamples = data.getN_s();
    for(int epoch = 0; epoch < numEpochs; epoch++){
        cout << "starting epoch " << epoch << endl;
        for(int s = 0; s < numSamples; s++){

            // for each node i in the input layer get the activation
            for(int i = 0; i < N_i; i++){
                activations[0][i] = data.getFeature(s,i);
            }

            // for the non-input layers, forward propagate the activations
            for(int l = 1; l < 3; l++){
                // for each node j in the layer, update the activation using the previous layer
                for(int j = 0; j < activations[l].size(); j++){
                    layerSum[l][j] = weights[l-1][0][j] * -1;
                    // for each node k in the previous layer, add the weighted activation to the current node
                    for(int k = 0; k < activations[l-1].size(); k++){
                        layerSum[l][j] += weights[l-1][k+1][j] * activations[l-1][k];
                    }
                    activations[l][j] = SIG(layerSum[l][j]);
                }
            }

            // propagate deltas backward from output layer
            // for each node o in the output layer, get the delta
            for(int o = 0; o < N_o; o++){
                deltas[2][o] = SIGD(layerSum[2][o]) * (data.getLabel(s,o) - activations[2][o]);
            }

            // go back through the layers and calculate delta for each node
            // only 1 hidden layer, so no looping through layers. Just update hidden layer
            // for each node h in layer update the delta
            for (int h = 0; h < deltas[1].size(); h++){
                tmp = 0;
                // loop through next layer to get sum of the deltas * weights
                for (int o = 0; o < deltas[2].size(); o++){
                    tmp += weights[1][h+1][o] * deltas[2][o];
                }
                deltas[1][h] = SIGD(layerSum[1][h]) * tmp;
            }

            // loop through each weight in the network and apply the update equation
            for(int l = 0; l < weights.size(); l++){
                for(int i = 0; i < weights[l].size()-1; i++){
                    for(int j = 0; j < weights[l][i+1].size(); j++){
                        weights[l][i+1][j] += deltas[l+1][j] * activations[l][i] * learnRate;
                        if (i == 0)
                            weights[l][0][j] -= learnRate * deltas[l+1][j];
                    }
                }
            }
        }
    }
}

void NeuralNet::test(Dataset &data, string outfile){

    // verify the dataset matches the neural net
    if (N_i != data.getN_i() || N_o != data.getN_o()){
        cerr << "ERROR: Dataset and Network size mismatch\n";
        exit(1);
    }
    int numSamples = data.getN_s();

    // initialize variables
    vector< vector<double> > activations(3,vector<double>(1,0));
    activations[0].resize(N_i);
    activations[1].resize(N_h);
    activations[2].resize(N_o);
    vector< vector<double> > layerSum(activations);

    vector< vector<double> > results(N_o,vector<double>(4,0));

    // for each sample run the neural net
    for (int s = 0; s < numSamples; s++){

        // load data into input layer
        for(int i = 0; i < N_i; i++){
            activations[0][i] = data.getFeature(s,i);
        }

        for(int l = 1; l < 3; l++){
            // for each node j in the layer, update the activation using the previous layer
            for(int j = 0; j < activations[l].size(); j++){
                //layerSum[l][j] = 0;
                layerSum[l][j] = weights[l-1][0][j] * -1;
                // for each node k in the previous layer, add the weighted activation to the current node
                for(int k = 0; k < activations[l-1].size(); k++){
                    layerSum[l][j] += weights[l-1][k+1][j] * activations[l-1][k];
                }
                activations[l][j] = SIG(layerSum[l][j]);
            }
        }

        // store the results and update metrics
        for(int o = 0; o < N_o; o++){
            activations[2][o] = round(activations[2][o]);
            if (data.getLabel(s,o)){
                if (activations[2][o])
                    results[o][A]++;
                else
                    results[o][C]++;
            }
            else {
                if (activations[2][o])
                    results[o][B]++;
                else
                    results[o][D]++;
            }
        }
    }

    // print results
    double a = 0, a_tot = 0;
    double b = 0, b_tot = 0;
    double c = 0, c_tot = 0;
    double d = 0, d_tot = 0;
    double acc = 0, micro_acc = 0, macro_acc = 0;   //accuracy
    double prec = 0, micro_prec = 0, macro_prec = 0;//precision
    double rec = 0, micro_rec = 0, macro_rec;       //recall
    double f1 = 0, micro_f1 = 0, macro_f1 = 0;      //f1
    string sp(" ");

    ofstream fileout;
    fileout.open(outfile);
    fileout.setf(ios::fixed,ios::floatfield);

    // print stats for each output
    for (int o = 0; o < N_o; o++){
        a = results[o][A];
        b = results[o][B];
        c = results[o][C];
        d = results[o][D];

        a_tot += a;
        b_tot += b;
        c_tot += c;
        d_tot += d;

        acc = (a+d)/(a+b+c+d);
        prec = a/(a+b);
        rec = a/(a+c);
        f1 = (2*prec*rec)/(prec+rec);

        macro_acc += acc;
        macro_prec += prec;
        macro_rec += rec;

        fileout.precision(0);
        fileout << a << sp << b << sp << c << sp << d;

        fileout.setf(ios::fixed,ios::floatfield);
        fileout.precision(3);
        fileout << sp << acc << sp << prec << sp << rec << sp << f1 << endl;
    }

    // calculate micro averages
    micro_acc = (a_tot+d_tot)/(a_tot+b_tot+c_tot+d_tot);
    micro_prec = a_tot/(a_tot+b_tot);
    micro_rec = a_tot/(a_tot+c_tot);
    micro_f1 = (2*micro_prec*micro_rec)/(micro_prec+micro_rec);

    fileout << micro_acc << sp << micro_prec << sp << micro_rec << sp << micro_f1 << endl;

    // calculate macro averages
    macro_acc /= N_o;
    macro_prec /= N_o;
    macro_rec /= N_o;
    macro_f1 = (2*macro_prec*macro_rec)/(macro_prec+macro_rec);

    fileout << macro_acc << sp << macro_prec << sp << macro_rec << sp << macro_f1 << endl;
}

// load a training set
bool Dataset::load(string infile){
    ifstream filein;
    filein.open(infile);
    string instring;

    // read dataset params
    filein >> instring;
    N_s = stoi(instring);
    filein >> instring;
    N_i = stoi(instring);
    filein >> instring;
    N_o = stoi(instring);

    // resize vectors
    features.resize(N_s);
    labels.resize(N_s);
    for(int s = 0; s < N_s; s++){
        features[s].resize(N_i);
        labels[s].resize(N_o);
    }

    // populate data vectors
    for(int s = 0; s < N_s; s++){
        for (int i = 0; i < N_i; i++){
            filein >> instring;
            features[s][i] = stod(instring);
        }
        for(int o = 0; o < N_o; o++){
            filein >> instring;
            labels[s][o] = stod(instring);
        }
    }
}
