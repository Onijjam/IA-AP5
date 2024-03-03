//
// Created by Gauthier Montagne on 19/02/2024.
//
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "Lecture/Bmp2Matrix.h"
#include "functions.h"

// Redéfinir la fonction relu
float relu(float x) {
    if (x < 0) {
        return 0;
    }
    return x;
}

// Redéfinir la fonction softmax
void softmax(float *input, int length) {
    float m = -INFINITY;
    for (int i = 0; i < length; i++) {
        if (input[i] > m) {
            m = input[i];
        }
    }

    float sum = 0.0;
    for (int i = 0; i < length; i++) {
        sum += expf(input[i] - m);
    }

    float offset = m + logf(sum);
    for (int i = 0; i < length; i++) {
        input[i] = expf(input[i] - offset);
    }
}

// Adapter la fonction flatten
void flatten(unsigned char **image, float flattenedImage[]) {
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            flattenedImage[i * 28 + j] = (float)image[i][j];
        }
    }
}

// Fonction pour charger les poids à partir d'un fichier
void loadWeights(float *weights, const char *weightsPath, int sizeWeights, int sizeBias) {
    FILE *pfichierWeights = fopen(weightsPath, "rb");
    if (pfichierWeights == NULL) {
        printf("Erreur lors de l'ouverture du fichier de poids.\n");
        exit(EXIT_FAILURE);
    }

    // Lecture des poids à partir du fichier
    for (int i = 0; i < sizeWeights * sizeBias; i++) {
        fscanf(pfichierWeights, "%f", &weights[i]);
    }

    fclose(pfichierWeights);
}

// Fonction pour charger les biais à partir d'un fichier
void loadBiases(float *biases, const char *biasesPath, int sizeBias) {
    FILE *pfichierBiases = fopen(biasesPath, "rb");
    if (pfichierBiases == NULL) {
        printf("Erreur lors de l'ouverture du fichier de biais.\n");
        exit(EXIT_FAILURE);
    }

    // Lecture des biais à partir du fichier
    for (int i = 0; i < sizeBias; i++) {
        fscanf(pfichierBiases, "%f", &biases[i]);
    }

    fclose(pfichierBiases);
}


void dense_relu(Layer *layer, int input_size, int output_size, float *input, float **output) {
    // Allocation de mémoire pour la sortie
    *output = (float *)malloc(output_size * sizeof(float));

    // Vérification de l'allocation de mémoire
    if (*output == NULL) {
        fprintf(stderr, "Erreur lors de l'allocation de mémoire pour output\n");
        exit(1);
    }

    printf("memoire ok\n");

    // Calcul de la sortie de la couche dense avec ReLU
    for (int i = 0; i < output_size; i++) {
        // Initialisation de la sortie avec le biais correspondant
        (*output)[i] = layer->biases[i];

        // Calcul de la somme pondérée des entrées
        for (int j = 0; j < input_size; j++) {
            // Correction de l'accès aux poids
            (*output)[i] += layer->weights[j][i] * input[j];
        }

        // Application de ReLU à la sortie
        (*output)[i] = relu((*output)[i]);
    }
}

void dense_softmax(Layer *layer, int input_size, int output_size, float *input, float **output) {
    // Allocation de mémoire pour la sortie
    *output = (float *)malloc(output_size * sizeof(float));

    // Vérification de l'allocation de mémoire
    if (*output == NULL) {
        fprintf(stderr, "Erreur lors de l'allocation de mémoire pour output\n");
        exit(1);
    }

    // Calcul de la sortie de la couche dense avec softmax
    for (int i = 0; i < output_size; i++) {
        // Initialisation de la sortie avec le biais correspondant
        (*output)[i] = layer->biases[i];

        // Calcul de la somme pondérée des entrées
        for (int j = 0; j < input_size; j++) {
            (*output)[i] += layer->weights[i][j] * input[j];
        }
    }

    // Application de softmax à la sortie
    softmax(*output, output_size);
}