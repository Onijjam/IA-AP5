#include <stdio.h>
#include <stdlib.h>
#include "functions.h"
#include "Lecture/Bmp2Matrix.h"

int main(int argc, char* argv[]) {
    // Chargement des poids et des biais à partir des fichiers
    char weightsPath1[100];
    char biasesPath1[100];
    sprintf(weightsPath1, "./../weights/layer_1_weights.txt");
    sprintf(biasesPath1, "./../biases/layer_1_biases.txt");
    int inputSize = 784; // Taille de l'entrée
    int outputSize1 = 1176; // Taille de la sortie après la première couche dense
    int outputSize2 = 10; // Taille de la sortie après la deuxième couche dense

    // Couche 1
    float *bias1 = (float *)malloc(outputSize1 * sizeof(float));
    float *weights1 = (float *)malloc(inputSize * outputSize1 * sizeof(float));
    loadWeights(weights1, weightsPath1, inputSize, outputSize1);
    loadBiases(bias1, biasesPath1, outputSize1);
    layer1.biases = bias1;
    layer1.weights = (float **)malloc(outputSize1 * sizeof(float *));
    for (int i = 0; i < outputSize1; i++) {
        layer1.weights[i] = &weights1[i * inputSize];
    }

    // Couche 2
    char weightsPath2[100];
    char biasesPath2[100];
    sprintf(weightsPath2, "./../weights/layer_2_weights.txt");
    sprintf(biasesPath2, "./../biases/layer_2_biases.txt");

    float *bias2 = (float *)malloc(outputSize2 * sizeof(float));
    float *weights2 = (float *)malloc(outputSize1 * outputSize2 * sizeof(float));
    loadWeights(weights2, weightsPath2, outputSize1, outputSize2);
    loadBiases(bias2, biasesPath2, outputSize2);
    layer2.biases = bias2;
    layer2.weights = (float **)malloc(outputSize2 * sizeof(float *));
    for (int i = 0; i < outputSize2; i++) {
        layer2.weights[i] = &weights2[i * inputSize];
    }

    printf("Weights and biases loaded successfully\n");

    // Tableau pour compter les prédictions correctes par classe
    int correct_predictions[10] = {0};

    // Boucle pour traiter chaque image dans le dossier "Images"
    for (int digit = 0; digit <= 9; digit++) {
        for (int index = 0; index < 10; index++) {
            // Chargement de l'image
            BMP bitmap;
            char filePath[100];
            sprintf(filePath, "./../Images/%d_%d.bmp", digit, index);
            FILE *pFichier = fopen(filePath, "rb");

            if (pFichier == NULL) {
                printf("Erreur dans la lecture du fichier %s\n", filePath);
                return 1;
            }
            LireBitmap(pFichier, &bitmap);
            fclose(pFichier);

            ConvertRGB2Gray(&bitmap);

            // Aplatir l'image
            float flatImage[inputSize];
            flatten(bitmap.mPixelsGray, flatImage);

            // Appliquer une première couche dense suivie d'une fonction d'activation ReLU
            float *output1 = NULL;
            dense_relu(&layer1, inputSize, outputSize1, flatImage, &output1);

            // Appliquer la deuxième couche dense suivie d'une fonction d'activation softmax
            float *output2 = NULL;
            dense_softmax(&layer2, outputSize1, outputSize2, output1, &output2);

            // Trouver la classe ayant le taux de confiance le plus élevé
            int predicted_class = 0;
            float max_confidence = output2[0];

            for (int i = 1; i < outputSize2; i++) {
                if (output2[i] > max_confidence) {
                    max_confidence = output2[i];
                    predicted_class = i;
                }
            }

            // Vérifier si la prédiction est correcte et mettre à jour le tableau des prédictions correctes
            if (digit == predicted_class) {
                correct_predictions[digit]++;
            }

            // Libérer la mémoire allouée dynamiquement
            free(output1);
            free(output2);

            // Libérer la mémoire allouée dynamiquement pour l'image
            DesallouerBMP(&bitmap);
        }
    }

// Calcul du nombre total d'images
    int total_images = 10 * 10; // 10 classes, 10 images par classe

// Calcul du nombre total de prédictions correctes
    int total_correct_predictions = 0;
    for (int i = 0; i < 10; i++) {
        total_correct_predictions += correct_predictions[i];
    }

// Calcul du pourcentage de prédictions correctes
    float accuracy_percentage = (float)total_correct_predictions / total_images * 100;

// Affichage des résultats
    printf("Total correct predictions: %d\n", total_correct_predictions);
    printf("Total images: %d\n", total_images);
    printf("Accuracy: %.2f%%\n", accuracy_percentage);

// Libérer la mémoire allouée dynamiquement pour les couches
    free(layer1.biases);
    free(layer1.weights);

    free(layer2.biases);
    free(layer2.weights);

    return 0;

}
