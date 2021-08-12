# Leveraging effectiveness and efficiency in Page Stream Deep Segmentation
**Engineering Applications of Artificial Intelligence**

*Volume 105, October 2021, 104394*

https://doi.org/10.1016/j.engappai.2021.104394

## Code Organization

*  `TOBACCO800` - jupyter notebooks and supporting python classes for tobacco800 dataset experiments
*  `ailab_splitter` - jupyter notebooks and supporting python classes for AI Lab Splitter dataset experiments

## Data

[Tobacco 800 and AI Lab Splitter](https://onedrive.live.com/?authkey=%21AGSL%2DQed7EsCmoQ&id=1C53FEF72A2AC3F3%2123947&cid=1C53FEF72A2AC3F3)

## Citation

Fabricio Ataides Braz, Nilton Correia da Silva, Jonathan Alis Salgado Lima,
Leveraging effectiveness and efficiency in Page Stream Deep Segmentation,
Engineering Applications of Artificial Intelligence,
Volume 105,
2021,
104394,
ISSN 0952-1976,
https://doi.org/10.1016/j.engappai.2021.104394.
(https://www.sciencedirect.com/science/article/pii/S0952197621002426)

### Latex

@article{BRAZ2021104394,
title = {Leveraging effectiveness and efficiency in Page Stream Deep Segmentation},
journal = {Engineering Applications of Artificial Intelligence},
volume = {105},
pages = {104394},
year = {2021},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2021.104394},
url = {https://www.sciencedirect.com/science/article/pii/S0952197621002426},
author = {Fabricio Ataides Braz and Nilton Correia {da Silva} and Jonathan Alis Salgado Lima},
keywords = {Page Stream Segmentation, Classification},
abstract = {The separation of documents contained in a page stream is a critical activity in some segments. That is the case of the Brazilian judiciary system since it is overwhelmed with files resulting from batch scanning of lawsuits, finishing in PDFs containing several types of mixed documents. To make such a file usable, we must divide it into cohesive sets of pages that result in a single piece. The typical approach to this task involves sorting the page into a stream that reveals the transition between documents. That is, it is about identifying the page that highlights a new piece in the stream. For this task, classification methods combining text and image got the best results, obtaining accuracy and kappa scores respectively of 91.9% and 83.1% in the Tobacoo800 dataset. This outcome, although remarkable, requires excessive computational demand. In this work, by changing the entry of image models and employing a novel labeling system, we achieved the same result, without the overhead that the modal text imposes on the solution. In addition, we built a new public dataset called AI.Lab.Splitter specifically aimed at page stream segmentation task with more than 30k labeled samples. Finally, in addition to VGG, we used EfficientNet, whose number of parameters is 1/6 of the former. We could observe an advantage close to 2.5% in f1 score, compared to the same proposal using VGG in our dataset.}
}

