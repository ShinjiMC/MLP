#include <iostream>
#include "mlp.hpp"
#include "dataset.hpp"

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "Use: ./main [dataset.txt]" << std::endl;
        return 1;
    }

    std::string dataset_file = argv[1];
    Dataset dataset(dataset_file);
    std::vector<std::vector<float>> X = dataset.get_X();
    std::vector<float> y_scalar = dataset.get_y();

    if (X.empty() || y_scalar.empty())
    {
        std::cerr << "Error: Dataset is empty." << std::endl;
        return 1;
    }
    std::vector<std::vector<float>> Y;
    for (float v : y_scalar)
        Y.push_back({v});
    int n_inputs = X[0].size();
    std::vector<std::function<float(float)>> activations = {sigmoid, relu};
    std::vector<std::function<float(float)>> derivatives = {sigmoid_derivative, relu_derivative};
    std::vector<int> layers = {n_inputs, 2, 1};
    MLP mlp(layers, activations, derivatives, 0.01f);
    std::cout << "Entrenando MLP..." << std::endl;
    mlp.train(X, Y, 0.0001f, false, dataset_file);
    std::cout << "Predicciones:\n";
    for (size_t i = 0; i < X.size(); ++i)
    {
        std::vector<float> pred = mlp.predict(X[i]);
        std::cout << "(";
        for (float val : X[i])
            std::cout << val << " ";
        std::cout << ") => " << pred[0] << " ≈ " << (pred[0] > 0.5f ? 1 : 0) << " (esperado: " << Y[i][0] << ")\n";
    }

    return 0;
}
