<div id="toc">
   <ul align="center" style="list-style: none;">
  <a href="https://github.com/Nokia-Bell-Labs/efficient_layer_selection"><img src="assets/system.png"></a>
  <summary>
     <h1>AdaBet</h1> <br>
    <h2>Gradient-free Layer Selection for Efficient Training of Deep Neural Networks</h2>
  </summary>
   </ul>
</div>

## :rocket: Updates
- Oct 2nd 2025: The code for AdaBet is available on [GitHub](https://github.com/Nokia-Bell-Labs/efficient_layer_selection).

## :book: Summary
To utilize pre-trained neural networks on edge and mobile devices, we often require efficient adaptation to user-specific runtime data distributions while operating under limited compute and memory resources. On-device retraining with a target dataset can facilitate such adaptations; however, it remains impractical due to the increasing depth of modern neural nets, as well as the computational overhead associated with gradient-based optimization across all layers. Current approaches reduce training cost by selecting a subset of layers for retraining, however, they rely on labeled data, at least one full-model backpropagation, or server-side meta-training; limiting their suitability for constrained devices. We introduce AdaBet, a gradient-free layer selection approach to rank important layers by analyzing topological features of their activation spaces through Betti Numbers and using forward passes alone. AdaBet allows selecting layers with high learning capacity, which are important for retraining and adaptation, without requiring labels or gradients. Evaluating AdaBet on sixteen pairs of benchmark models and datasets, shows AdaBet achieves an average gain of 5% more classification accuracy over gradient-based baselines while reducing average peak memory consumption by 40%.