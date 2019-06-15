# phos

TODO: scrsht

----

#### 1. Growing toolbox of "interesting" neural network layers.

#### 2. Environment for comparing and debugging their performance.

#### 3. Different nn.Module API to fix edge case, PyTorch wrappers (phos.nn for torch.nn).

----

### Creating a benchmark

Evaluate the models on a set of hyperparameters (dataset, model depth/width, optimizer, etc).

```
python3 -m phos.bench.new --settings phos/bench/example.json --dir data/benchmark/example/
```

### Updating a benchmark

Add new layers you invent to `phos.nx`, subclassing `phos.nx.base.layer.Layer`.

Add new models demoing such layers to `phos.model`, calling `register_model` with your block(s) to plug into the standard model frame for comparison ([example](https://github.com/knighton/phos/blob/master/phos/model/baseline.py)).

Update a pre-existing benchmark, executing any new models found.

```
python3 -m phos.bench.add_models --dir data/benchmark/example/
```

### Viewing a benchmark

Analyze model performance, pathways, losses, etc.

```
python3 -m phos.bench.view --dir data/benchmark/example/ --port 1337
```

----

### Phos module

* Skip/choice connections are common (framework was built to see into this).

* Layers altering the loss in interesting ways is also common.

* Combined, it's quite ugly to scale auxiliary losses to account for dynamically weighting different information pathways, as opposed to blithely catting onto the loss without balancing, like some peasant.

* Phos solves by adding an aux loss parameter to Module.forward and scaling it together with the data.

* To handle lab introspection, phos modules also have a method blurb() which recursively summarizes automatic shape/dtype/latency statistics as well as custom information like path weight to JSON.  And so on.

----

*Elemental phosphorus was first isolated (as white phosphorus) in 1669 and emitted a faint glow when exposed to oxygen – hence the name, taken from Greek mythology, Φωσφόρος meaning "light-bearer".*
