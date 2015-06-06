package geym.nn;

import java.util.List;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.core.transfer.Linear;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.nnet.comp.neuron.ThresholdNeuron;
import org.neuroph.nnet.learning.PerceptronLearning;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.LayerFactory;
import org.neuroph.util.NeuralNetworkFactory;
import org.neuroph.util.NeuralNetworkType;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.random.NguyenWidrowRandomizer;

public class CompositePerceptron extends NeuralNetwork <LearningRule> {

	private static final long serialVersionUID = 9002409230079313865L;

	public CompositePerceptron(List<Integer> neuronsInLayers, NeuronProperties neuronProperties) {
		this.createNetwork(neuronsInLayers, neuronProperties);
	}

	private void createNetwork(List<Integer> neuronsInLayers, NeuronProperties neuronProperties) {

		// set network type
		this.setNetworkType(NeuralNetworkType.MULTI_LAYER_PERCEPTRON);

		// create input layer
		NeuronProperties inputNeuronProperties = new NeuronProperties(InputNeuron.class, Linear.class);
		Layer layer = LayerFactory.createLayer(neuronsInLayers.get(0), inputNeuronProperties);

		boolean useBias = true; // use bias neurons by default
		if (neuronProperties.hasProperty("useBias")) {
			useBias = (Boolean) neuronProperties.getProperty("useBias");
		}

		if (useBias) {
			layer.addNeuron(new BiasNeuron());
		}

		this.addLayer(layer);

		// create layers
		Layer prevLayer = layer;

		// for(Integer neuronsNum : neuronsInLayers)
		int layerIdx =1;
		for (layerIdx = 1; layerIdx < neuronsInLayers.size()-1; layerIdx++) {
			Integer neuronsNum = neuronsInLayers.get(layerIdx);
			// createLayer layer
			layer = LayerFactory.createLayer(neuronsNum, neuronProperties);

			if (useBias && (layerIdx < (neuronsInLayers.size() - 1))) {
				layer.addNeuron(new BiasNeuron());
			}

			// add created layer to network
			this.addLayer(layer);
			// createLayer full connectivity between previous and this layer
			if (prevLayer != null) {
				ConnectionFactory.fullConnect(prevLayer, layer);
			}

			prevLayer = layer;
		}
		
		Integer neuronsNum = neuronsInLayers.get(layerIdx);
		NeuronProperties lineProperties = new NeuronProperties(ThresholdNeuron.class, Linear.class);
		lineProperties.setProperty("thresh", 0.5);
		layer = LayerFactory.createLayer(neuronsNum, lineProperties);
		this.addLayer(layer);
		if (prevLayer != null) {
			ConnectionFactory.fullConnect(prevLayer, layer);
		}
		
		
		// set input and output cells for network
		NeuralNetworkFactory.setDefaultIO(this);

		// set learnng rule
		 this.setLearningRule(new PerceptronLearning());
		//this.setLearningRule(new MomentumBackpropagation());
		// this.setLearningRule(new DynamicBackPropagation());

		this.randomizeWeights(new NguyenWidrowRandomizer(-0.7, 0.7));

	}
}
