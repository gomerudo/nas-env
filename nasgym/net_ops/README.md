# Notes for development

## FIXME: Crutial

- [ ] For `net_builder.py`. Check why BatchNormalization is giving more outputs in graph, which were not indicated in the code.
- [ ] For `net_builder.py`. Check if `Add` node needs padding too: design tests with maxpooling and no-padding.
- [ ] For `net_trainer.py`. When evaluation is called after training, the log of training is not shown in TensorBoard (see `nas-env/test/test_net_trainer.py`).

## FIXME: Minor

- [X] Order sequence by layer_index. Check feasibility.

## TODO

### `net_builder.py`

- [X] Finish the final concatenation.
- [ ] Currently, the network is built given two main assumptions: a) the layers are ordered by `layer_index`, and b) the predecesors of the layer are always lower than the layer's index. The first assumption should not cause any special issues when not satisfied cause we can order the layers, however if the second assumption is not satisfied then there will be errors cause the predecesor is not yet included. We need to think of a solution. Ideas: 1) Make the network non-valid and return `reward=0`, 2) Previosuly to the building process, try to optimize the architecture representation algorithmically (this can be **too much work**). I think the second option is encouraged cause otherwise the agent will ignore some architectures (note that not all the time we can optimize the arechitecture representation, but maybe sometimes will be). **ANALYZE EVERYTHING STATED HERE**.
- [ ] In general, think about how to handle the non-valid architectures. My first idea is to return `reward=0`.
- [ ] Check if we can move the layer codes to a global scope in `__init__.py`

### `net_trainer.py`

- [ ] For `net_trainer.py`. See how to implement the learning rate decay used in BlockQNN.\
- [ ] Implement the distributed strategy (if I remember correctly, the mirrored one). This will imply some changes to the shape of the input, as mentioned in the documentation. Follow [this tutorial](https://www.tensorflow.org/guide/distribute_strategy).
- [ ] Implement the Eearly Stop Strategy implemented by BlockQNN, which uses the FLOPS and other net's property.
- [ ] Make the training procedure to log the useful information, such as accuracy, loss and any other thing that can be consider useful.
- [ ] Try to improve the feeding of the data via the `input_fn`.
