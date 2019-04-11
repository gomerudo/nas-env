# Notes for development

## FIXME: Crutial

- [ ] For `net_builder.py`. Check why BatchNormalization is giving more outputs in graph, which were not indicated in the code.
- [X] For `net_builder.py`. Check if `Add` node needs padding too: design tests with maxpooling and no-padding.

## FIXME: Minor

- [ ] For `net_trainer.py`. See how to implement the learning rate decay used in BlockQNN.\
- [ ] Try to improve the feeding of the data via the `input_fn`.

## TODO

### `net_builder.py`

- [X] Currently, the network is built given two main assumptions: a) the layers are ordered by `layer_index`, and b) the predecesors of the layer are always lower than the layer's index. The first assumption should not cause any special issues when not satisfied cause we can order the layers, however if the second assumption is not satisfied then there will be errors cause the predecesor is not yet included. We need to think of a solution. Ideas: 1) Make the network non-valid and return `reward=0`, 2) Previosuly to the building process, try to optimize the architecture representation algorithmically (this can be **too much work**). I think the second option is encouraged cause otherwise the agent will ignore some architectures (note that not all the time we can optimize the arechitecture representation, but maybe sometimes will be). **ANALYZE EVERYTHING STATED HERE**.
- [X] Remove the dependency on the sorting for the building of the network.

### `net_trainer.py`

- [ ] Implement the distributed strategy (if I remember correctly, the mirrored one). This will imply some changes to the shape of the input, as mentioned in the documentation. Follow [this tutorial](https://www.tensorflow.org/guide/distribute_strategy).
