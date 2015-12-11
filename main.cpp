#include <regex>

#include "nn.h"

using namespace std;

void train_network(){
    NeuralNet a;
    Dataset d;
    string instring;
    double learnRate;
    int numEpochs;

    cout << "enter the name of your neural net file: ";
    cin >> instring;
    a.load(instring);

    cout << "enter the name of your training file: ";
    cin >> instring;
    d.load(instring);

    cout << "enter the learning rate: ";
    cin >> instring;
    learnRate = stod(instring);

    cout << "enter the number of Epochs: ";
    cin >> instring;
    numEpochs = stoi(instring);

    cout << "enter the name of the output file: ";
    cin >> instring;

    cout << "start training function\n";
    a.train(d,learnRate,numEpochs);
    cout << "training completed. saving output to file\n";
    a.save(instring);
}

void test_network(){

}

int main(){
	string instring;

	while(1) {
		cout << "would you like to train or test? (train/test)\n";
		cin >> instring;
		if (regex_match(instring,regex("train"))) {
			train_network();
			break;
		}
		if (regex_match(instring,regex("test"))) {
			test_network();
			break;
		}
		cout << "invalid input. try again.\n";
	}
    return 0;
}