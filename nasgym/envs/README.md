# Development notes

## FIXME: critic

## FIXME: minor

- [ ] Store the strings of the actions (`actions_info`) using the codes (`1`, `2` ...) instead of `convolution`, `remove`, etc.
- [ ] Currently, the logic is a bit dependent of the NSC code. For a more robust solution, we need to provide classes/interfaces so that users can implement their own environments/encodings/etc.
- [ ] In general, make the code here cleaner.
- [ ] For the `render()` function we are printing the state, but it could also be nice to make a plot of the network (this should come from the `net_ops` or/and `utl` modules).

## TODO

- [ ] Decide on the actions to support
- [X] If the `nasenv.yml` is not found, create the default configuration with the content we have now. Ideas: create a function to return the hardcoded string of the YAML file. **Withdrawn**.
