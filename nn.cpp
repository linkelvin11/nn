#include "nn.h"

// define logistic sigmoid and its derivative
#define SIG(x) 1/(1+exp(-x))
#define SIGD(x) SIG(x)*(1-SIG(x))

using namespace std;

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
            fileout << weights[0][i][h] << ' ';
        }
        fileout << endl;
    }

    for(int o = 0; o < N_o; o++){
        for(int h = 0; h < N_h+1; h++){
            fileout << weights[1][h][o] << ' ';
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

            //cout << "initialize input layer for sample " << s+1 << endl;
            // for each node i in the input layer get the activation
            for(int i = 0; i < N_i; i++){
                activations[0][i] = data.getFeature(s,i);
            }

            //cout << "forward propagate the activations\n";
            // for the non-input layers, forward propagate the activations
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

            //cout << "calculate deltas for output layer\n";
            // propagate deltas backward from output layer
            // for each node o in the output layer, get the delta
            for(int o = 0; o < N_o; o++){
                deltas[2][o] = SIGD(layerSum[2][o]) * (data.getLabel(s,o) - activations[2][o]);
            }

            //cout << "back propagate the deltas from the output layer\n";
            // go back through the layers and calculate delta for each node
            // only 1 hidden layer, so no looping through layers. Just update hidden layer
            // for each node h in layer update the delta
            for (int h = 0; h < deltas[1].size()-1; h++){
                tmp = 0;
                // loop through next layer to get sum of the deltas * weights
                for (int o = 0; o < deltas[2].size(); o++){
                    tmp += weights[1][h+1][o] * deltas[2][o];
                }
                deltas[1][h] = SIGD(layerSum[1][h]) * tmp;
            }

            // cout << "update each weight\n";
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

    this->save("trainedoutput.txt");


}

void NeuralNet::test(Dataset &data){

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

    vector< vector<double> > results;
    results.resize(numSamples);

    double a = 0;
    double b = 0;
    double c = 0;
    double d = 0;

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
            //cout << activations[2][0] << " ";;
            activations[2][o] = round(activations[2][o]);
            if (data.getLabel(s,o)){
                if (activations[2][o])
                    a++;
                else
                    c++;
            }
            else {
                if (activations[2][o])
                    b++;
                else
                    d++;
            }
        }
        //cout << endl;

        results[s] = activations[2];
    }

    cout << "overall accuracy: " << (a+d)/(numSamples) << endl;
    cout << a << endl;
    cout << c << endl;
    cout << b << endl;
    cout << d << endl;

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

// save a testing set
bool Dataset::save(string outfile){

}

int main(){
    NeuralNet a;
    a.load(string("wdbc.init"));
    //a.save(string("save.txt"));
    Dataset d;
    d.load("wdbc.train");
    a.train(d,0.1,100);
    a.save("100epochstrained.txt");

    Dataset tester;
    tester.load("wdbc.test");
    a.test(tester);
    return 0;
}