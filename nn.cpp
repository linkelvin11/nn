#include "nn.h"

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
    cout << N_i << N_h << N_o << endl;

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
            cout << weights[0][i][h] << ' ';
        }
        cout << endl;
    }
    for (int o = 0; o < N_o; o++){
        for (int h = 0; h < N_h+1; h++){
            filein >> instring;
            weights[1][h][o] = stod(instring);
            cout << weights[1][h][o]<< ' ';
        }
        cout << endl;
    }

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

}

int main(){
    NeuralNet a;
    a.load(string("test.txt"));
    a.save(string("save.txt"));
    return 0;
}