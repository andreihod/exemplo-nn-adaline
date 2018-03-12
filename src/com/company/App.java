package com.company;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class App {

    private final static double FATOR = 0.1;

    private double[][] weights = new double[3][3];
    private double[] biasWeight = new double[3];
    private double[] dwBias = new double[3];

    public App() {

        Random rnd = new Random();

        double[][] datasetIn = readDataset("./in_FN.txt");
        double[][] datasetOut = readDataset("./out_FN.txt");

        int total = new Double(datasetIn.length * 0.3).intValue();
        double[][] datasetInTrain = new double[datasetIn.length - total][3];
        double[][] datasetOutTrain = new double[datasetIn.length - total][3];
        double[][] datasetInTest = new double[total][3];
        double[][] datasetOutTest = new double[total][3];
        for (int i = 0; i < datasetIn.length; i++) {
            if (i < datasetIn.length - total) {
                datasetInTrain[i] = datasetIn[i];
                datasetOutTrain[i] = datasetOut[i];
            } else {
                datasetInTest[i - (datasetIn.length - total)] = datasetIn[i];
                datasetOutTest[i - (datasetIn.length - total)] = datasetOut[i];
            }
        }

        System.out.println("Pesos iniciais");
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights.length; j++) {
                double w = (Math.random()*20)-10;
                System.out.println("["+i+"]"+"["+j+"]: " + w + " ");
                weights[i][j] = w;
            }
        }

        double sumE[] = new double[]{1.0,1.0,1.0};
        int epochs = 0;
        while (epochs < 1000000) {
            double[][] dw = new double[3][3];
            sumE[0] = 0.0;
            sumE[1] = 0.0;
            sumE[2] = 0.0;
            dwBias[0] = 0.0;
            dwBias[1] = 0.0;
            dwBias[2] = 0.0;
            int cont = 0;
            double erroTotal = 0.0;
            for(double[] lineIn : datasetInTrain) {
                sumE[0] += this.fit(0, lineIn, datasetOutTrain[cont][0], dw[0]);
                sumE[1] += this.fit(1, lineIn, datasetOutTrain[cont][1], dw[1]);
                sumE[2] += this.fit(2, lineIn, datasetOutTrain[cont][2], dw[2]);
                erroTotal += Math.abs(sumE[0]);
                erroTotal += Math.abs(sumE[1]);
                erroTotal += Math.abs(sumE[2]);
                cont++;
            }
            for (int x = 0; x < weights.length; x++) {
                for (int dwi = 0; dwi < dw.length; dwi++) {
                    weights[x][dwi] += dw[x][dwi];
                }
                biasWeight[x] += dwBias[x];
            }
            //this.printEpoch(sumE, weights);
            epochs++;
            if (erroTotal == 0.0) {
                break;
            }
        }
        System.out.println();
        System.out.println(epochs + " epocas");

        System.out.println("Iniciando testes...");

        int acertos = 0;
        for (int i = 0; i < datasetInTest.length; i++) {
            int contAcertos = 0;
            for (int j = 0; j < datasetInTest[i].length; j++) {
                double outPredicted = this.executa(j, datasetInTest[i]);
                if(outPredicted == datasetOutTest[i][j]) {
                    contAcertos++;
                }
                //System.out.println("Linha ["+ i +"]["+ j +"]: " +
                //        + outPredicted + " Valor original: " + datasetOutTest[i][j]);
            }
            if(contAcertos==3) {
                acertos++;
            }
        }

        System.out.println("Acertou " + acertos + " de " + datasetInTest.length);
        System.out.println("Acurácia de " + new Double(acertos) / datasetInTest.length);

    }

    private void printEpoch(double[] sumE, double[][] weights) {
        for (int x = 0; x < sumE.length; x++) {
            System.out.print("error: " + sumE[x] + "");
            for (int i = 0; i < weights[x].length; i++) {
                System.out.print(" w["+x+"]:" + weights[x][i]);
            }
            System.out.println("");
        }
    }

    public double fit(int index, double[] in, double out, double[] dw) {
        double error = out - this.executa(index, in);
        for (int i = 0; i < in.length; i++) {
            dw[i] += (in[i] * error * FATOR);
        }
        this.dwBias[index] += (1 * error * FATOR);
        return error;
    }

    public int executa(int index, double[] in) {
        double sum = 0.0;
        for (int i = 0; i < in.length; i++) {
            sum += in[i] * weights[index][i];
        }
        sum += 1 * biasWeight[index];
        return this.threshold(sum);
    }

    public int threshold(double sum) {
        return sum > 0 ? 1 : 0;
    }

    public double[][] readDataset(String path) {
        double[][] dataset;
        try {
            List<String> lines = Files.readAllLines(Paths.get(path));
            dataset = new double[lines.size()][3];
            int cont = 0;
            for(String line : lines) {
                String[] arrayLine = line.split(";");
                dataset[cont][0] = Double.parseDouble(arrayLine[0]);
                dataset[cont][1] = Double.parseDouble(arrayLine[1]);
                dataset[cont][2] = Double.parseDouble(arrayLine[2]);
                cont++;
            }
            return dataset;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

}
