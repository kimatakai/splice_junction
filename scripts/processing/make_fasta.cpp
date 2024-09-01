#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

int main() {
    std::ifstream infile("splice_junction_data.tsv");  // Input file
    std::ofstream outfile("splice_junction.fasta");  // Output file in FASTA format
    std::string line;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string type, id, sequence;

        // Parse the line
        iss >> type >> id >> sequence;

        // Write to the FASTA file
        outfile << ">" << id << std::endl;
        outfile << sequence << std::endl;
    }

    infile.close();
    outfile.close();

    return 0;
}