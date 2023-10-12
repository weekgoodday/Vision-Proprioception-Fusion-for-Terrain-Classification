# A CNN Based Vision-Proprioception Fusion Method for Robust UGV Terrain Classification
a pytorch implementation of 2021 RAL paper [A CNN Based Vision-Proprioception Fusion Method for Robust UGV Terrain Classification](https://ieeexplore.ieee.org/abstract/document/9507312)

The dataset is downloaded from [open access](https://ieee-dataport.org/open-access/jackal-robot-7-class-terrain-dataset-vision-and-proprioception-sensors)

## Accuracy
### Image Net Part 
after training 5 epochs on trainSet_c7_corrected.hdf5,
| Experiment | Backbone | Testset | Dark | Sim_fog | Sim_sun |
| :-: | :-----------: | :------: | :------: | :------: | :------: |
| 1 | MobileNetV2 | **99.63%** | 53.27% | 68.98% | 65.72% | 
| 2 | MobileNetV2 | **99.93%** | 21.22% | 26.57% | 37.59% | 
| 3 | MobileNetV2 | **99.70%** | 25.15% | 54.95% | 47.84% | 