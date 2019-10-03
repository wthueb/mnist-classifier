#ifndef NET_H
#define NET_H

#include <cassert>
#include <cstring>
#include <random>
#include <vector>

class Net
{
public:
    Net() = delete;
    
    Net(uint32_t input, uint32_t hidden, uint32_t output)
        : m_input_size(input), m_hidden_size(hidden), m_output_size(output)
    {
        // bias neurons so + 1
        m_input_neurons.resize(m_input_size + 1);
        m_hidden_neurons.resize(m_hidden_size + 1);

        m_output_neurons.resize(m_output_size);

        memset(m_input_neurons.data(), 0, m_input_neurons.size() * sizeof(double));
        memset(m_hidden_neurons.data(), 0, m_hidden_neurons.size() * sizeof(double));
        memset(m_output_neurons.data(), 0, m_output_neurons.size() * sizeof(double));

        m_input_neurons.back() = -1.0;
        m_hidden_neurons.back() = -1.0;

        m_input_hidden_weights.resize(m_input_neurons.size() * m_hidden_neurons.size());
        m_hidden_output_weights.resize(m_hidden_neurons.size() * m_output_neurons.size());

        init_weights();
    }

    void init_weights()
    {
        std::default_random_engine gen;
        std::normal_distribution<double> dist(0.0, 1.0);

        for (auto input = 0u; input < m_input_size; input++)
            for (auto hidden = 0u; hidden < m_hidden_size; hidden++)
                m_input_hidden_weights[input*m_hidden_size + hidden] = dist(gen);

        for (auto hidden = 0u; hidden < m_hidden_size; hidden++)
            for (auto output = 0u; output < m_hidden_size; output++)
                m_input_hidden_weights[hidden*m_output_size + output] = dist(gen);
    }

    std::vector<double>& forward(const std::vector<double>& input)
    {
        assert(input.size() == m_input_size);
        assert(m_input_neurons.back() == -1.0 && m_hidden_neurons.back() == -1.0);

        memcpy(m_input_neurons.data(), input.data(), input.size() * sizeof(double));

        for (auto hidden = 0u; hidden < m_hidden_size; hidden++)
        {
            m_hidden_neurons[hidden] = 0;

            for (auto input = 0u; input <= m_input_size; input++)
            {
                auto idx = input*m_hidden_size + hidden;
                m_hidden_neurons[hidden] += m_input_neurons[input]*m_input_hidden_weights[idx];
            }

            m_hidden_neurons[hidden] = std::max<double>(0.0, m_hidden_neurons[hidden]);
        }

        for (auto output = 0u; output < m_output_size; output++)
        {
            m_output_neurons[output] = 0;

            for (auto hidden = 0u; hidden <= m_hidden_size; hidden++)
            {
                auto idx = hidden*m_output_size + output;
                m_output_neurons[output] += m_hidden_neurons[hidden]*m_hidden_output_weights[idx];
            }

            m_output_neurons[output] = std::max<double>(0.0, m_output_neurons[output]);
        }

        return m_output_neurons;
    }

    uint32_t m_input_size;
    uint32_t m_hidden_size;
    uint32_t m_output_size;

    std::vector<double> m_input_neurons;
    std::vector<double> m_input_hidden_weights;
    std::vector<double> m_hidden_neurons;
    std::vector<double> m_hidden_output_weights;
    std::vector<double> m_output_neurons;
};

#endif // NET_H
