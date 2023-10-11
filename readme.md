# A CNN Based Vision-Proprioception Fusion Method for Robust UGV Terrain Classification
a pytorch implementation of 2021 RAL paper [A CNN Based Vision-Proprioception Fusion Method for Robust UGV Terrain Classification](https://ieeexplore.ieee.org/abstract/document/9507312)

The dataset is downloaded from [open access](https://ieee-dataport.org/open-access/jackal-robot-7-class-terrain-dataset-vision-and-proprioception-sensors)

## Accuracy
### Image Net Part 
after training 5 epochs on trainSet_c7_corrected.hdf5,
| Backbone | Testset | Dark | Sim_fog | Sim_sun |
| -------- | ------ | ------ | ------ | ------ |
| MobileNetV2 | 99.55% | xx% | xx% | xx% | 