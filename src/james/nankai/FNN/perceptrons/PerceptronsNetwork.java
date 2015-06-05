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
		//�����������BiasNeuron,��ʾ��Ԫƫ��
		inputLayer.addNeuron(new BiasNeuron());
		//���亯����Step
		NeuronProperties outputNeuronProperties = new NeuronProperties();
		outputNeuronProperties.setProperty("transferFunction", TransferFunctionType.STEP);
		//����㣬Ҳ������Ԫ
		Layer outputLayer = LayerFactory.createLayer(1,outputNeuronProperties);
		this.addLayer(outputLayer);
		//�����������뵼����Ԫ
		ConnectionFactory.fullConnect(inputLayer,outputLayer);
		NeuralNetworkFactory.setDefaultIO(null);
		//���ø�֪��ѧϰ�㷨��LMSѧϰ�㷨
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













