# Notes for development

## FIXME: Crutial

- [ ] Check why BatchNormalization is giving more outputs in graph, which were not indicated in the code.
- [ ] Check if `Add` node needs padding too: design tests with maxpooling and no-padding.

## FIXME: Minor

- [ ] Order sequence by layer_index.


## TODO

- [X] Implement training of network: make a different method that takes the net as argument and implements the training procedure.
- [X] Set the parameters used in BlockQNN for the training. **Missing the learning rate decay**.
- [X] Check how to distribute the nodes on the different gpu devices, on-the-fly: i.e. at running time, see which nodes are free, and use it. Ideas: if checking availability a the moment is not feasible, maybe keep a set of lock files with the assigned nodes, when finished, freeze the lock.

