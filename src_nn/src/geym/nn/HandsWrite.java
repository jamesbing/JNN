package geym.nn;

import geym.nn.bmp.BmpReader;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;

public class HandsWrite  implements LearningEventListener{

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		new HandsWrite().run();
	}

	public void run() throws IOException {
    	
        DataSet trainingSet = new DataSet(256, 10);
        
        int trainNum=3;
        for(int i=0;i<trainNum;i++){
	        for(int j=0;j<10;j++){
		        String f=String.format("handswriter\\train%d\\%d.bmp", i,j);
		        double[] bNum=BmpReader.convertBmp2Inputs(f);
		        double[] bRe=BmpReader.convertBmp2Outputs(f);
		        trainingSet.addRow(new DataSetRow(bNum, bRe));
	        }
        }
        
        // create multi layer perceptron
        MultiLayerPerceptron myMlPerceptron = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, 256, 128, 10);
        myMlPerceptron.getLearningRule().setMaxError(0.0001d);
        // enable batch if using MomentumBackpropagation
        if( myMlPerceptron.getLearningRule() instanceof MomentumBackpropagation )
        	((MomentumBackpropagation)myMlPerceptron.getLearningRule()).setBatchMode(true);

        LearningRule learningRule = myMlPerceptron.getLearningRule();
        learningRule.addListener(this);
        
        // learn the training set
        System.out.println("Training neural network...");
        myMlPerceptron.learn(trainingSet);

        // test perceptron
        System.out.println("Testing trained neural network");
        testNeuralNetwork(myMlPerceptron, trainingSet);

    }
    public static void testNeuralNetwork(NeuralNetwork neuralNet, DataSet testSet) throws IOException {
        for(int i=0;i<10;i++){
	        String f=String.format("handswriter\\test\\%d.bmp", i);
	        double[] bNum=BmpReader.convertBmp2Inputs(f);
	        neuralNet.setInput(bNum);
	        neuralNet.calculate();
	        double[] networkOutput = neuralNet.getOutput();
	        System.out.print("Input: " + i);
	        System.out.println(" Output: " + Arrays.toString( networkOutput) );
        }
  
    }
    
    @Override
    public void handleLearningEvent(LearningEvent event) {
        BackPropagation bp = (BackPropagation)event.getSource();
        System.out.println(bp.getCurrentIteration() + ". iteration : "+ bp.getTotalNetworkError());
    }  
}
