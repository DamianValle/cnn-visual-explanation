# Visual Explanation for Weakly Supervised Object Localization

Code for my [thesis](https://findit.dtu.dk/en/catalog/2450517180) carried out at [DTU](http://www.dtu.dk).
We weakly train a CNN classifier with the [rico dataset](http://interactionmining.org/rico) on existence of icons in mobile app screenshots. 

<img src="/examples/rico.jpg" width="250"/>

We then compute the activation maximization of the input test images to figure out the input pixels responsible for the activations of each output neuron.

<img src="/examples/18original_pred1.png" width="250"/> <img src="/examples/18heat_pred1.png" width="250"/> 

After that we perform [OTSU thresholding](http://ijarcet.org/wp-content/uploads/IJARCET-VOL-2-ISSUE-2-387-389.pdf) to get prediction candidates and blob analysis to choose the most likely correct prediction.

<img src="/examples/18thres_pred1.png" width="250"/> <img src="/examples/18thresbb_pred1.png" width="250"/> 

![Menu Activation](/examples/menu-activation.png)
