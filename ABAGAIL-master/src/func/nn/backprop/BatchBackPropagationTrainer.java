package func.nn.backprop;

import shared.DataSet;
import shared.GradientErrorMeasure;
import shared.Instance;
import func.nn.NetworkTrainer;

import java.text.DecimalFormat;

/**
 * A standard batch back propagation trainer
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class BatchBackPropagationTrainer extends NetworkTrainer {
    
    /**
     * The weight update rule to use
     */
    private WeightUpdateRule rule;
    private DataSet train;
    private DataSet test;
    private static DecimalFormat df = new DecimalFormat("0.000");
    private int iterations = 1;
    
    /**
     * Make a new back propagation trainer
     * @param patterns the patterns to train on
     * @param network the network to train
     * @param errorMeasure the error measure to use
     */
    public BatchBackPropagationTrainer(DataSet patterns, 
            BackPropagationNetwork network, 
            GradientErrorMeasure errorMeasure,
            WeightUpdateRule rule) {
        super(patterns, network, errorMeasure);
        this.rule = rule;
    }

    public BatchBackPropagationTrainer(DataSet patterns,
                                       BackPropagationNetwork network,
                                       GradientErrorMeasure errorMeasure,
                                       WeightUpdateRule rule, DataSet train, DataSet test) {
        super(patterns, network, errorMeasure);
        this.rule = rule;
        this.train = train;
        this.test = test;
    }

    public double train() {
        BackPropagationNetwork network =
                (BackPropagationNetwork) getNetwork();
        GradientErrorMeasure measure =
                (GradientErrorMeasure) getErrorMeasure();
        DataSet patterns = train;
        double trainError = 0;
        for (int i = 0; i < patterns.size(); i++) {
            Instance pattern = patterns.get(i);
            network.setInputValues(pattern.getData());
            network.run();
            Instance output = new Instance(network.getOutputValues());
            double[] errors = measure.gradient(output, pattern);
            trainError += measure.value(output, pattern);
            network.setOutputErrors(errors);
            network.backpropagate();
        }
        //System.out.println(error/(patterns.size()));
        //errStr += (error/(patterns.size())) + " ";

        patterns = test;
        double testError = 0;
        for (int i = 0; i < patterns.size(); i++) {
            Instance pattern = patterns.get(i);
            network.setInputValues(pattern.getData());
            network.run();
            Instance output = new Instance(network.getOutputValues());
            double[] errors = measure.gradient(output, pattern);
            testError += measure.value(output, pattern);
            network.setOutputErrors(errors);
            network.backpropagate();
        }
        //System.out.println(testError/(patterns.size()));
        //testErrStr += (testError/(patterns.size())) + " ";
        System.out.println( df.format("Iteration " + String.format("%04d" ,iterations++) + ": " + trainError / (double) 398) + " " + df.format(testError / (double) 171));
        network.updateWeights(rule);
        network.clearError();
        return trainError / patterns.size();
    }
    

}
