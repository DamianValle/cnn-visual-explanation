# Visual Explanation for Weakly Supervised Object Localization

Code for my [thesis](https://findit.dtu.dk/en/catalog/2450517180) carried out at [DTU](http://www.dtu.dk).
We weakly train a CNN classifier with the [rico dataset](http://interactionmining.org/rico) on existence of icons in mobile app screenshots. 

![Rico dataset](/examples/rico.jpg "Rico dataset")

We then compute the class activation mapping by implementing the [Grad-CAM algorithm](https://arxiv.org/abs/1610.02391) of the input test images to figure out the input pixels responsible for the activations of each output neuron.

<img src="/examples/18original_pred1.png" width="250"/> <img src="/examples/18heat_pred1.png" width="250"/> 

After that we perform [OTSU thresholding](http://ijarcet.org/wp-content/uploads/IJARCET-VOL-2-ISSUE-2-387-389.pdf) to get prediction candidates and blob analysis to choose the most likely correct prediction.

<img src="/examples/18thres_pred1.png" width="250"/> <img src="/examples/18thresbb_pred1.png" width="250"/>

Finally we iteratively perform activation maximization to see what the trained network has learned. Example outputs for the arrow and menu classes.

![House](/examples/house-activation.png "House activations")
![Love](/examples/love-activation.png "Love activations")
