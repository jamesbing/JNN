package geym.nn;

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

public class Parity  implements LearningEventListener{

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		new Parity().run();
	}
	
	public static double[] int2double(int i){
		double[] re=new double[32];
		for(int j=0;j<32;j++){
			re[j]=(double)((i>>j)&1);
		}
		return re;
	}
	
	/**
	 * 0001  正偶数
	 * 0010  负偶数
	 * 0100  正奇数
	 * 1000  负奇数
	 * @param i
	 * @return
	 */
	public static double[] int2prop(int i){
		double[] pe={0d,0d,0d,1d};
		double[] ne={0d,0d,1d,0d};
		double[] po={0d,1d,0d,0d};
		double[] no={1d,0d,0d,0d};
		if(i>0 && i%2==0){
			return pe;
		}else if(i<0 && i%2==0){
			return ne;
		}else if(i>0 && i%2!=0){
			return po;
		}else if(i<0 && i%2!=0){
			return no;
		}
		return pe;
	}
	 /**
     * Runs this sample
     */
    public void run() {
    	
        // create training set (logical XOR function)
        DataSet trainingSet = new DataSet(32, 4);
        for(int i=0;i<2000;i++){
        	int in=new Random().nextInt();
        	trainingSet.addRow(new DataSetRow(int2double(in), int2prop(in)));
        }
       
        // create multi layer perceptron
        MultiLayerPerceptron myMlPerceptron = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, 32, 10, 4);
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
    public static void testNeuralNetwork(NeuralNetwork neuralNet, DataSet testSet) {

    	for(int i=0;i<100;i++){
    		int in=new Random().nextInt();
    		double[] inputnumber=int2double(in);
    		neuralNet.setInput(inputnumber);
    		neuralNet.calculate();
    	    double[] networkOutput = neuralNet.getOutput();
    	    
    	    System.out.print("Input: " + in);
    	    System.out.println(" Output: " + Arrays.toString( networkOutput) );
    	}
    }
    
    @Override
    public void handleLearningEvent(LearningEvent event) {
        BackPropagation bp = (BackPropagation)event.getSource();
        System.out.println(bp.getCurrentIteration() + ". iteration : "+ bp.getTotalNetworkError());
    }   
}

