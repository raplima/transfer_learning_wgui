# User interface for transfer learning applications

I reworked the scripts for fitting and evaluation. I also reworked some plots and added more options for metrics. This version uses pysimplegui for the interface part. 
There is the possibility to modify the top model and the optimizers using the interface now, but typing is involved (I tried to keep good defaults and whoever wants to test more stuff will have to type). Currently only dense and dropout layers are available for top model modification. 

Some comparisons using remote sensing data and transfer learning here https://www.mdpi.com/2072-4292/12/1/86/htm might be useful:

```Bibtex
@article{PiresdeLima2019,
  doi = {10.3390/rs12010086},
  url = {https://doi.org/10.3390/rs12010086},
  year = {2019},
  month = dec,
  publisher = {{MDPI} {AG}},
  volume = {12},
  number = {1},
  pages = {86},
  author = {Rafael Pires de Lima and Kurt Marfurt},
  title = {Convolutional Neural Network for Remote-Sensing Scene Classification: Transfer Learning Analysis},
  journal = {Remote Sensing}
}
```

----

### Quick summary
* helper_functions.py: some data and model manipulation functions;
* plotting_functions.py: plotting functions (loss decay, confusion matrix);
* model_fit.py: fitting functions;
* model_evaluate.py: evaluate trained model 
* main.py: uses the scripts above, serve as example and also as better control for tests
* user_interface: the user interface, calls functions in scripts above through interface. 

