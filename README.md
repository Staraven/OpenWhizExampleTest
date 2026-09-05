# OpenWhizExamplesTest

A GlistEngine app that runs `libs/OpenWhiz`'s `OpenWhiz/text` module inside a
real GlistApp, porting its two standalone examples
(`examples/textClassificationExample`, `examples/clusterLabelingExample`)
from plain `std::cout` executables into `gCanvas.cpp` using `gLogi` and
GlistEngine's asset paths. `runTextClassificationExample()` runs the
tokenize -> stem -> embed -> classify pipeline in English, Turkish, and
French; `runClusterLabelingExample()` runs the tokenize -> TF-IDF ->
unsupervised-cluster -> bag-of-stems-label pipeline against the embedded
20 Newsgroups sample (`assets/files/twenty_ng_sample.txt`) and logs the
same purity measurement as the standalone example.

To run it: clone this repo under `glist/myglistapps` in a GlistEngine
installation, open it as a GlistApp project (Eclipse/CLion/whatever your
GlistEngine setup uses), build, and run — output appears in the log console
on startup (`gCanvas::setup()`), the render window itself just shows the
GlistEngine logo.
