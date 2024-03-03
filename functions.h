//
// Created by Gauthier Montagne on 19/02/2024.
//

#ifndef IA_FUNCTIONS_H
#define IA_FUNCTIONS_H

typedef struct {
    float *biases;
    float **weights;
} Layer;

Layer layer1;
Layer layer2;



void flatten(unsigned char **image, float flattenedImage[]);
float relu(float x);
void softmax(float *input, int size);
void loadWeights(float *weights, const char *weightsPath, int sizeWeights, int sizeBias);
void loadBiases(float *biases, const char *biasesPath, int sizeBias);
void dense_softmax(Layer *layer, int input_size, int output_size, float *input, float **output) ;
void dense_relu(Layer *layer, int input_size, int output_size, float *input, float **output);

#endif //IA_FUNCTIONS_H
