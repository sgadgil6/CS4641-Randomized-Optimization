package opt.test;

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;
import func.nn.feedfwd.FeedForwardNetwork;
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import opt.RandomizedHillClimbing;
import shared.*;
import shared.filt.RandomOrderFilter;
import shared.filt.TestTrainSplitFilter;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class BPBreastCancer {
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 5, outputLayer = 1, trainingIterations = 500;
    private static FeedForwardNeuralNetworkFactory factory = new FeedForwardNeuralNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    //private static FeedForwardNetwork networks[] = new FeedForwardNetwork[100];
    private static FeedForwardNetwork networks[] = new FeedForwardNetwork[3];
    //private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[100];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    //private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[100];
    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    //private static String[] oaNames = new String[100];
    //private static String[] oaNames = new String[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";
    private static List<List<Double>> oaResultsTrain = new ArrayList<>();
    private static List<List<Double>> oaResultsTest = new ArrayList<>();

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {

        new RandomOrderFilter().filter(set);
        TestTrainSplitFilter ttsf = new TestTrainSplitFilter(70);
        ttsf.filter(set);
        DataSet train = ttsf.getTrainingSet();
        DataSet test = ttsf.getTestingSet();

        BackPropagationNetworkFactory factory2 =
                new BackPropagationNetworkFactory();
        BackPropagationNetwork network2 = factory2.createClassificationNetwork(
                new int[] {inputLayer, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, outputLayer});
        FixedIterationTrainer trainer = new FixedIterationTrainer(
                new BatchBackPropagationTrainer(train, network2,
                        new SumOfSquaresError(), new RPROPUpdateRule(), train, test), trainingIterations);
        double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
        trainer.train();
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);

        double predicted, actual;
        start = System.nanoTime();
//        for(int j = 0; j < train.getInstances().length; j++) {
//            network2.setInputValues(train.getInstances()[j].getData());
//            network2.run();
//            actual = -1;
//            String[] labelArr = train.getInstances()[j].getLabel().toString().split(", ");
//            for (int k = 0; k < labelArr.length; k++) {
//                if (Double.parseDouble(labelArr[k]) == 1.0) {
//                    actual = k;
//                }
//            }
//            labelArr = network2.getOutputValues().toString().split(", ");
//            predicted = Double.parseDouble(labelArr[(int)actual]);
//            double trash = Math.abs(predicted) > .1 ? correct++ : incorrect++;
//        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        results +=  "\nTRAINING: Results for " + "backprop" + ": \nCorrectly classified " + correct + " instances." +
                "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        System.out.println(results);
    }


    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[569][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/breastCancerDataset/breastCancer_train.csv")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");
                attributes[i] = new double[2][];
                attributes[i][0] = new double[5]; // 5 attributes
                attributes[i][1] = new double[1];
//                System.out.println(scan.nextLine());
                for(int j = 0; j < 5; j++) {
                    attributes[i][0][j] = Float.parseFloat(scan.next());
                }

                attributes[i][1][0] = Float.parseFloat(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
    }
}
