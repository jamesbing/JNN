package james.nankai.FNN.perceptrons;

import org.neuroph.core.Layer;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.nnet.learning.LMS;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.LayerFactory;
import org.neuroph.util.NeuralNetworkFactory;
import org.neuroph.util.NeuralNetworkType;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;

public class PerceptronsNetwork {

	
	private NeuralNetworkType networkType;
	
	private void createNetwork(int inputNeuronsCount){
		//set the type of the network is perceptronsNetwork
		this.setNetworkType(NeuralNetworkType.PERCEPTRON);
		//setup input neuranons, this is used to demonstrate the input stimulation
		NeuronProperties inputNeuronProperties = new NeuronProperties();
		inputNeuronProperties.setProperty("neuronType",InputNeuron.class);
		Layer inputLayer = LayerFactory.createLayer(inputNeuronsCount,inputNeuronProperties);
		this.addLayer(inputLayer);
		//在输入层增加BiasNeuron,表示神经元偏置
		inputLayer.addNeuron(new BiasNeuron());
		//传输函数是Step
		NeuronProperties outputNeuronProperties = new NeuronProperties();
		outputNeuronProperties.setProperty("transferFunction", TransferFunctionType.STEP);
		//输出层，也就是神经元
		Layer outputLayer = LayerFactory.createLayer(1,outputNeuronProperties);
		this.addLayer(outputLayer);
		//将输入层的输入导向神经元
		ConnectionFactory.fullConnect(inputLayer,outputLayer);
		NeuralNetworkFactory.setDefaultIO(null);
		//设置感知机学习算法，LMS学习算法
		this.setLearningRule(new LMS());
		
	}
	
	/**TODO:setNetworkType function
		james
	 */
	private void setLearningRule(LMS lms) {
		// TODO Auto-generated method stub
		
	}

	/**TODO:setNetworkType function
		james
	*/
	public void setNetworkType(NeuralNetworkType T){
		this.networkType = T;
	}
	
	/**TODO:addLayer function
		james
	 */
	public void	addLayer(Layer layer){
		
	}
	
	
	
}













