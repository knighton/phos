# phos

TODO: scrsht

Features
--------

1. Growing toolbox of "interesting" neural network layers.

2. Environment for comparing and debugging their performance.

3. The torch nn.Module forward API is defective.  Fixed in phos.  Breaks compat.  Old-style modules can be trivially wrapped (phos.nn for torch.nn).  See section below.

Workflow
--------

1. Create a "lab".  Easy enough to type.

```
python3 -m phos.lab.new --settings phos/lab/example.json --dir data/lab/example/
```

2. Invent some architecture.

3. Plug the modules into a model chassis used for comparison.

4. Fit model/s against the lab's hyperparameter grid.  Get coffee.

```
python3 -m phos.lab.update --dir data/lab/example/
```

5. Analyze its performance, pathways, losses, etc.

```
python3 -m phos.lab.view --dir data/lab/example/ --port 1337
```

Modules
-------

* Skip/choice connections are common (framework was built to see into this).

* Layers altering the loss in interesting ways is also common.

* Combined, it's quite ugly to scale auxiliary losses to account for dynamically weighting different information pathways, as opposed to blithely catting onto the loss without balancing, like some peasant.

* Phos solves by adding an aux loss parameter to Module.forward and scaling it together with the data.

* To handle lab introspection, phos modules also have a method blurb() which recursively summarizes automatic shape/dtype/latency statistics as well as custom information like path weight to JSON.  And so on.

Name
----

*Elemental phosphorus was first isolated (as white phosphorus) in 1669 and emitted a faint glow when exposed to oxygen – hence the name, taken from Greek mythology, Φωσφόρος meaning "light-bearer".*
