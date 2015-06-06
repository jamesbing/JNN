package geym.nn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.core.learning.SupervisedLearning;
import org.neuroph.nnet.comp.neuron.ThresholdNeuron;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;

public class FunctionAppro  implements LearningEventListener{

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		new FunctionAppro().run();
	}
	
	public static double[] int2double(int i){
		double[] re=new double[32];
		for(int j=0;j<32;j++){
			re[j]=(double)((i>>j)&1);
		}
		return re;
	}
	
	public static int double2int(double[] d){
		int re=0;
		for(int i=0;i<d.length;i++){
			if(d[i]>0.5){
				re|=(1<<i);
			}
		}
		return re;
	}
	
	 /**
     * Runs this sample
     */
    public void run() {
    	
        // create training set (logical XOR function)
        DataSet trainingSet = new DataSet(1, 1);
        for(int i=0;i<2000;i++){
        	double in=new Random().nextDouble()*4-2;
        	//double out=1+Math.sin(Math.PI/4*in);
        	double out=in;
        	trainingSet.addRow(new DataSetRow(new double[]{in}, new double[]{out}));
        }
        List<Integer> count=new ArrayList<Integer>();
        count.add(1);
        count.add(16);
        count.add(1);
        NeuronProperties neuronProperties = new NeuronProperties();
        neuronProperties.setProperty("useBias", false);
        neuronProperties.setProperty("transferFunction", TransferFunctionType.SIGMOID);
        neuronProperties.setProperty("neuronType", ThresholdNeuron.class);
        neuronProperties.setProperty("thresh", 0.5);
        // create multi layer perceptron
        CompositePerceptron myMlPerceptron = new CompositePerceptron(count,neuronProperties);
        // enable batch if using MomentumBackpropagation
       // if( myMlPerceptron.getLearningRule() instanceof MomentumBackpropagation )
        //	((MomentumBackpropagation)myMlPerceptron.getLearningRule()).setBatchMode(true);

        LearningRule learningRule = myMlPerceptron.getLearningRule();
        learningRule.addListener(this);
        ((SupervisedLearning)myMlPerceptron.getLearningRule()).setMaxError(0.0001d);
        // learn the training set
        System.out.println("Training neural network...");
        myMlPerceptron.learn(trainingSet);

        // test perceptron
        System.out.println("Testing trained neural network");
        testNeuralNetwork(myMlPerceptron, trainingSet);

    }
    public static void testNeuralNetwork(NeuralNetwork neuralNet, DataSet testSet) {

    	for(int i=0;i<100;i++){
    		double in=new Random().nextDouble()*4-2;
    		neuralNet.setInput(in);
    		neuralNet.calculate();
    	    double[] networkOutput = neuralNet.getOutput();
    	    
    	    System.out.print("Input: " + in);
    	    System.out.println(" Output: " + networkOutput[0] );
    	}
    }
    
    @Override
    public void handleLearningEvent(LearningEvent event) {
    	SupervisedLearning bp = (SupervisedLearning)event.getSource();
        System.out.println(bp.getCurrentIteration() + ". iteration : "+ bp.getTotalNetworkError());
    }   
}

