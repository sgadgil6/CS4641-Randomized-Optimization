package func.nn.backprop;

import shared.*;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Scanner;

/**
 * A simple classification test
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class BatchBackULTrainer {


    private final static int inputLayer = 4, hiddenLayer = 1, outputLayer = 3, numRows = 4535;
    private static final String fileLoc = "src/opt/test/2016_New_Coder_Survey_NNRO_Normalized.csv";


    /**
     * Tests out the perceptron with the classic xor test
     * @param args ignored
     */
    public static void main(String[] args) {
        BackPropagationNetworkFactory factory =
                new BackPropagationNetworkFactory();
        Instance[] instances = initializeInstances();
        BackPropagationNetwork network = factory.createClassificationNetwork(
                new int[] { inputLayer, hiddenLayer, outputLayer});
        DataSet set = new DataSet(instances);
        FixedIterationTrainer trainer = new FixedIterationTrainer(
                new BatchBackPropagationTrainer(set, network,
                        new SumOfSquaresError(), new RPROPUpdateRule()), 1000);
        trainer.train();



        int correct = 0, incorrect = 0;
        double predicted, actual;
        for(int j = 0; j < instances.length; j++) {
            network.setInputValues(instances[j].getData());
            network.run();


            actual = -1;
            String[] labelArr = instances[j].getLabel().toString().split(", ");
            for (int k = 0; k < labelArr.length; k++) {
                if (Double.parseDouble(labelArr[k]) == 1.0) {
                    actual = k;
                }
            }
            labelArr = network.getOutputValues().toString().split(", ");
            predicted = 0;
            double winner = Double.parseDouble(labelArr[0]);
            for (int k = 1; k < labelArr.length; k++) {
                if (Double.parseDouble(labelArr[k]) > winner) {
                    predicted = k;
                    winner = Double.parseDouble(labelArr[k]);
                }
            }
            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        System.out.println("Accuracy: " + correct/((incorrect + correct)*1.0));
    }


    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[numRows][][]; // Number of rows

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File(fileLoc)));
            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[inputLayer]; // 7 attributes
                attributes[i][1] = new double[1];


                for(int j = 0; j < inputLayer; j++) {
                    String nextDub = scan.next();
                    attributes[i][0][j] = Double.parseDouble(nextDub);
                }
                String nextDub = scan.next();
                attributes[i][1][0] = Double.parseDouble(nextDub);
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);

            // Read the digit 0-9 from the attribute array that was read from the csv
            int c = (int) attributes[i][1][0];

            // Create a double array of length 10, all values are initialized to 0
            double[] classes = new double[outputLayer];

            // Set the i'th index to 1.0
            classes[c] = 1.0;
            instances[i].setLabel(new Instance(classes));
        }

        return instances;
    }
}
