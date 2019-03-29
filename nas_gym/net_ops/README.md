# Notes for development

## FIXME: Crutial

- [ ] Check why BatchNormalization is given more outputs in graph, which were not indicated in the code
- [ ] Check if Add needs padding too: Do tests with maxpooling and no-padding

## FIXME: Minor

- [ ] Order sequence by layer_index


## TODO

- [ ] Implement training of network: make a different method that takes the net as argument and implements the training procedure with the parameters in the paper.
- [ ] Check how to distribute the nodes on the different gpu devices, on-the-fly: i.e. at running time, see which nodes are free, and use it. Ideas: if checking availability a the moment is not feasible, maybe keep a set of lock files with the assigned nodes, when finished, freeze the lock.