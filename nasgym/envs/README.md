# Development notes

## FIXME: critic

## FIXME: minor

- [ ] Store the strings of the actions (`actions_info`) using the codes (`1`, `2` ...) instead of `convolution`, `remove`, etc.
- [ ] Currently, the logic is a bit dependent of the NSC code. For a more robust solution, we need to provide classes/interfaces so that users can implement their own environments/encodings/etc.
- [ ] In general, make the code here cleaner.
- [ ] For the `render()` function we are printing the state, but it could also be nice to make a plot of the network (this should come from the `net_ops` or/and `utl` modules).
- [ ] If the `nasenv.yml` is not found, create the default configuration with the content we have now. Ideas: create a function to return the hardcoded string of the YAML file.

## TODO

- [X] Define the reward range properly: i.e., think if [0, inf) is correct.
- [X] Test if defining the reward range actually makes a difference. NO.
- [X] Verify if `max_steps` is actually needed. Quick answer: yes, just make sure that the logic of `done` is correctly separating `max_steps` from `max_layers`. The latter is because with the `remove` actions we can have more than 10 steps before completing the `max_layers`.
- [X] Create the `DatasetHandler` that will take care of switching tasks (e.g. for meta-dataset) and provding the correct `train-validation` split.
- [X] Define the content of the `info_dict`. Ideas: The history of actions.
- [X] Implement the `train_network()` function that will call the `net_trainer` module.
