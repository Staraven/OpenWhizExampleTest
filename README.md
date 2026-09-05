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

---

App project template for [GlistEngine](https://github.com/GlistEngine/GlistEngine)

Developers can clone this repository to initialize a new GlistEngine game project. More information can be found on the [readme](https://github.com/GlistEngine/GlistEngine/blob/main/README.md) page of GlistEngine repository.

Please clone this repo under glist/myglistapps in your GlistEngine installation directory.

<u>_For MacOS (XCode)_</u>:

- If you do not have Xcode, download and install before continuing the guide.

- After cloning into the right directory, navigate to `~/dev/glist/myglistapps/GlistApp/_macos` from your terminal.

- From here, run command:

  `sh generate_glistapp_xcode.sh macos`
  
  The IOS specific instructions can be found in gipIOS plugin. 

- When Xcode opens the project, click on ALL_BUILD where it says `ALL_BUILD > My Mac` at the top center of the Xcode screen and select `GlistApp`.

- Click on the run button at the top-left of the screen.
