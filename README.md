# Face Similarity Calculator

A simple GUI application that allows users to calculate the similarity between two faces and potentially infer perceptions of attractiveness and prosociality between individuals.

## Overview

This app provides an easy-to-use graphical interface for users to upload pairs of images and calculate their facial similarity scores. Inspired by the findings from the paper, "Objectively measured facial traits predict in-person evaluations of facial attractiveness and prosociality in speed-dating partners," this tool could be valuable for those interested in the study of facial attractiveness and relationship dynamics.

## Installation

To get started with the Face Similarity Calculator, clone the repository and install the required dependencies.

```bash
git clone https://github.com/AspadaX/Face-Similarity-Calculator.git
cd Face-Similarity-Calculator
pip install -r requirements.txt
```
or, you may want to use it with a conda environment:
```
```

## Usage

After installation, run the application from the command line:

```bash
python FaceIDSimilarityComputation.py
```

The GUI should be running at `0.0.0.0:7863`, allowing you to upload images and calculate their similarity.

### Testing Outcomes

- Similarity between my synthetic image and my personal image: `0.8928`
- Similarity between my personal image and a random public figure: `-0.0772`
- Similarity between A celeberity couple : `0.0365`, and a couple of `0.0795` on the exact same images (I tested it for a couple of times, however, it eventually stablized at around 0.07)
- Similarity between me and the female of the celeberity couple: `-0.0621` (I got to say that she's not my type)
- Similarity between my personal image and an ex (average from multiple tests): ~ `0.03`

These outcomes are illustrative examples and may vary with each run. Exciting to see how your case works!

## Features

- Load images from your local file system
- Display image previews within the application
- Calculate facial similarity scores between two images
- User-friendly GUI for an intuitive experience

## Contributing

We encourage community contributions to expand the app's capabilities and improve its accuracy. If you're interested in volunteering for development or would like to contribute by sharing your own data, please reach me out on my discord `https://discord.gg/7bmgzFkn`, WeChat `baoxinyu2007` or submit a pull request.

## License

This project is open-source and available under the MIT License. See the LICENSE file for more details.

---

[Paper Reference]

Amy A.Z. Zhao et al., "Objectively measured facial traits predict in-person evaluations of facial attractiveness and prosociality in speed-dating partners." School of Psychology, University of Queensland, Australia.

---

This app is meant for educational and research purposes only and should not be used for making personal decisions regarding relationships or any form of emotional or psychological advice.
