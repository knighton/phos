# phos

![screenshot](https://github.com/knighton/phos/raw/master/doc/bench_cifar10.jpg)

## Features

* Growing toolbox of "interesting" neural network layers.

* Environment for comparing and debugging their performance.

* Different module API to fix edge case in PyTorch nn.Module; wrappers (phos.nn for torch.nn).

## Benchmarks

### New

* Evaluate the models on a set of hyperparameters (dataset, model depth/width, optimizer, etc).

  ```
  python3 -m phos.bench.new --settings phos/bench/example.json --dir data/benchmark/example/
  ```

### Update

* Add new layers you invent to `phos.nx`, subclassing `phos.nx.module.Module`.

* Add new models demoing such layers to `phos.model`, calling `register_model` with your block(s) to plug into the standard model frame for comparison ([example](https://github.com/knighton/phos/blob/master/phos/model/baseline.py)).

* Update a pre-existing benchmark, executing any new models found.

  ```
  python3 -m phos.bench.update --dir data/benchmark/example/
  ```

### View

* Analyze model performance, pathways, losses, etc.

  ```
  python3 -m phos.bench.view --dir data/benchmark/example/ --port 1337
  ```

## Modules

* Skip/choice connections are common (framework was built to see into this).

* Layers altering the loss in interesting ways is also common.

* Combined, it's quite ugly to scale auxiliary losses to account for dynamically weighting different information pathways, as opposed to blithely catting onto the loss without balancing, like some peasant.

* Phos solves by adding an aux loss parameter to Module.forward and scaling it together with the data.

* To handle lab introspection, phos modules also have a method blurb() which recursively summarizes automatic shape/dtype/latency statistics as well as custom information like path weight to JSON.  And so on.

----

*Elemental phosphorus was first isolated (as white phosphorus) in 1669 and emitted a faint glow when exposed to oxygen – hence the name, taken from Greek mythology, Φωσφόρος meaning "light-bearer".*
