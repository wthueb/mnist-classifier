#ifndef SYNAPSE_H
#define SYNAPSE_H

#include <memory>

#include "Neuron.h"

class Synapse
{
public:
    Synapse(std::shared_ptr<Neuron> input, std::shared_ptr<Neuron> output)
        : input(input), output(output) {}
    
    void set_weight(double w)
    {
        weight = w;
    }
    
    double get_weight()
    {
        return weight;
    }
    
    std::shared_ptr<Neuron> get_input()
    {
        return input;
    }
    
    std::shared_ptr<Neuron> get_output()
    {
        return output;
    }
    
private:
    double weight;
    
    std::shared_ptr<Neuron> input;
    std::shared_ptr<Neuron> output;
};

#endif // SYNAPSE_H
