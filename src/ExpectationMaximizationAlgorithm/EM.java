package ExpectationMaximizationAlgorithm;

import java.util.*;

public class EM {
    private static final double T = 1.0 / 3.0;
    private static final double SIGMA = 1;
    private static final double EPSILON = 0.001;

    public static void main(String[] args) {
        Scanner console = new Scanner(System.in);

        // input the size of data set
        int n = console.nextInt();

        // input the data set
        double[] data = new double[n];
        for (int i = 0; i < n; i++) {
            data[i] = console.nextDouble();
        }

        // initialize theta to be minimum, maximum and median respectively
        Arrays.sort(data);
        double[] currTheta = new double[3];
        currTheta[0] = data[0];
        currTheta[1] = data[data.length / 2 - 1];
        currTheta[2] = data[data.length - 1];

        // to update 3 theta
        double[] nextTheta = new double[3];

        // to store all expectations for given data set
        double[][] expectations = new double[data.length][3];

        // display
        System.out.printf("i\t\tu_1\t\tu_2\t\tu_3\t\tLogLihood\n");
        System.out.printf("[1,]\t%.5f\t%.5f\t%.5f\t%.5f\n", currTheta[0], currTheta[1], currTheta[2], 0.0);

        // keep doing until diff is less than the epsilon (or converged)
        double difference = Double.POSITIVE_INFINITY;
        int it = 2;
        while (difference >= EPSILON) {
            EStep(expectations, currTheta, data);   // Expectation step
            double log = MStep(expectations, nextTheta, data);  // Maximization step

            // get the maximum of three results
            difference = Math.max(nextTheta[0] - currTheta[0], nextTheta[1] - currTheta[1]);
            difference = Math.max(difference, nextTheta[2] - currTheta[2]);

            // display current round
            System.out.printf("[%d,]\t", it);
            for (int i = 0; i < 3; i++) {
                currTheta[i] = nextTheta[i];
                System.out.printf("%.5f\t", nextTheta[i]);
            }
            System.out.printf("%.5f\n", log);
            it++;
        }

    }


    // Calculate the conditional probability according to the definition
    private static double conditionalProbability(double theta, double x) {
        return Math.exp(-Math.pow(x - theta, 2) / (2 * Math.pow(SIGMA, 2)));
    }

    // Expectation step; calculate all expectations
    private static void EStep(double[][] expectations, double[] currTheta, double[] data) {
        for (int i = 0; i < expectations.length; i++) {
            double f1 = conditionalProbability(currTheta[0], data[i]);
            double f2 = conditionalProbability(currTheta[1], data[i]);
            double f3 = conditionalProbability(currTheta[2], data[i]);

            double denominator = f1 + f2 + f3;

            expectations[i][0] = f1 / denominator;
            expectations[i][1] = f2 / denominator;
            expectations[i][2] = f3 / denominator;
        }
    }

    // Maximization step; return the log likelihood of given data set
    private static double MStep(double[][] expectations, double[] nextTheta, double[] data) {
        for (int i = 0; i < 3; i++) {
            double numerator = 0.0;
            double denominator = 0.0;
            for (int j = 0; j < data.length; j++) {
                numerator += expectations[j][i] * data[j];
                denominator += expectations[j][i];
            }
            nextTheta[i] = numerator / denominator;
        }
        return LogLihood(expectations, nextTheta, data);
    }

    // calculate the log likelihood
    private static double LogLihood(double[][] expectations, double[] nextTheta, double[] data) {
        double LogLihood = Math.log(T) * data.length - Math.log(2 * Math.pow(SIGMA, 2) * Math.PI) * (data.length / 2);
        for (int i = 0; i < 3; i++) {
            double temp = 0.0;
            for (int j = 0; j < data.length; j++) {
                temp += (Math.pow(data[j] - nextTheta[i], 2) * expectations[j][i] / (2 * Math.pow(SIGMA, 2)));
            }
            LogLihood -= temp;
        }
        return LogLihood;
    }
}
