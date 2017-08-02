import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.factory.Nd4j;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;
import java.util.ArrayList;

public class KBSPurchaseRegressor {
    private static final Logger LOGGER = LoggerFactory.getLogger(KBSPurchaseRegressor.class);
    private String data;
    private int randomSeed;

    private NormalizerMinMaxScaler normalizer;
    private MultiLayerNetwork model;

    public KBSPurchaseRegressor(String data) throws Exception {
        this.data = data;
        normalizer = new NormalizerMinMaxScaler(0, 1);
        this.init();
    }

    private DataSet[] splitTestData(DataSetIterator dsi, Double testSize) {
        int inTest = 0;
        int inTrain = 0;
        DataSet input = dsi.next();
        int _testSize = Double.valueOf(input.numExamples() * testSize).intValue();
        int splitSize = input.numExamples() - _testSize;

        INDArray train_features = Nd4j.create(splitSize, input.getFeatures().columns());
        INDArray train_outcomes = Nd4j.create(splitSize, input.numOutcomes());
        INDArray test_features  = Nd4j.create(_testSize, input.getFeatures().columns());
        INDArray test_outcomes  = Nd4j.create(_testSize, input.numOutcomes());

        for (int i = 0; i < input.numExamples(); i++) {
            DataSet D = input.get(i);
            if (i <= splitSize) {
                train_features.putRow(inTrain, D.getFeatures());
                train_outcomes.putRow(inTrain, D.getLabels());
                inTrain += 1;
            } else {
                test_features.putRow(inTest, D.getFeatures());
                test_outcomes.putRow(inTest, D.getLabels());
                inTest += 1;
            }
        }

        return new DataSet[] {
            new DataSet(train_features, train_outcomes),
            new DataSet(test_features, test_outcomes)
        };
    }

    private DataSet trainData;
    private DataSet testData;
    public void init() throws Exception {
        File baseDir = new ClassPathResource("").getFile();
        SequenceRecordReader trainReader = new CSVSequenceRecordReader(0, ",");
        trainReader.initialize(new NumberedFileInputSplit(baseDir.getAbsolutePath() + "/" + this.data, 0, 0));

        int batchSize = 1;
        DataSet[] dataSets = this.splitTestData(
            new SequenceRecordReaderDataSetIterator(trainReader, batchSize, -1, 1, true),
            0.33
        );

        trainData = dataSets[0];
        testData = dataSets[1];

        normalizer.fitLabel(true);
        normalizer.fit(trainData);
        normalizer.transform(trainData);
        normalizer.transform(testData);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(140)
            //best option for diminishing gradient
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .weightInit(WeightInit.RELU)
            .updater(Updater.ADAM).momentum(0.9)
            .learningRate(0.0015)
            .list()
            .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10)
                .build())
            .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY).nIn(10).nOut(1).build())
            .build();

        model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(20));

        int nEpochs = 200;
        for (int i = 0; i < nEpochs; i++) {
            model.fit(trainData);
            RegressionEvaluation evaluation = new RegressionEvaluation(1);

            INDArray features = testData.getFeatureMatrix();
            INDArray labels = testData.getLabels();
            INDArray predicted = model.output(features, false);

            evaluation.evalTimeSeries(labels, predicted);
            System.out.println(evaluation.stats());
        }

        model.rnnTimeStep(trainData.getFeatureMatrix());
    }

    public INDArray predict() {
        INDArray predicted = model.rnnTimeStep(testData.getFeatureMatrix());
        normalizer.revertLabels(predicted);

        return predicted;
    }

    private List<String> allPlates = new ArrayList<String>();
    public int calculateNoOfPlates(String plate1, String plate2) {
        if (allPlates.isEmpty()) {
            char[] alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".toCharArray();
            for (char a : alphabet) {
                for (char b : alphabet) {
                    for (int i = 0; i < 10; i++) {
                        for (int j = 0; j < 10; j++) {
                            for (int k = 0; k < 10; k++) {
                                for (char c: alphabet) {
                                    allPlates.add(
                                        "K" + a + b + " " + i + "" + "" + j + "" + k + "" + c
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        return Math.abs(allPlates.indexOf(plate1) - allPlates.indexOf(plate2));
    }
}
