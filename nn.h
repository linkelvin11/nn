#include "constants.h"

class NeuralNet {
	// count of nodes in each layer (input,hidden,output)
	int N_i, N_h, N_o;
	// store weights in a 3D vector (layers,i,j)
	std::vector< std::vector< std::vector<double> > > weights;

public:
	bool load(std::string infile);
	bool save(std::string outfile);
};