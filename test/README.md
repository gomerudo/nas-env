# Development notes

## FIXME: critic

## FIXME: minor

- [X] Verify why the logged information when applying `evaluate` after `train` is not showing train's information.

## TODO

### `test_default_nasenv.py`

- [ ] Include tests for the elements in the resulting environment, such as the types of actions, etc.
- [X] Make the `nasenv.yml` a resource for all tests.
- [ ] Make more tests with different action sets. The idea will be to test the most strange and dramatic cases we can think of, such as: wrong predecesors, unordered layers (caused by `remove` operations).
- [ ] Make tests to verify the behavior of the `done` flag in scenarios such as `remove` operations and `max_steps`. Also, verify that the returned `state` is correct.

### `test_net_builder.py`
- [X] Test building weir architectures: wrong predecesors, unordered layers (caused by `remove` operations). This is related to the #2 of previous subsection.
