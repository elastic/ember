This directory is provided as a courtesy.  It includes the MalConv model to which we compared to in https://arxiv.org/abs/1804.04637.

For more details about MalConv, please see (and cite) the [original paper](https://arxiv.org/abs/1710.09435).

```
Raff, Edward, et al. "Malware detection by eating a whole exe." arXiv preprint arXiv:1710.09435 (2017).
```

If you use the pre-trained weights or code in your work, we also ask that you please cite [our paper](https://arxiv.org/pdf/1804.04637.pdf) for the implementation of MalConv, as it differs in a few subtle ways from the original.

```
H. Anderson and P. Roth, "EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models”, in ArXiv e-prints. Apr. 2018.

@ARTICLE{2018arXiv180404637A,
  author = {{Anderson}, H.~S. and {Roth}, P.},
  title = "{EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models}",
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint = {1804.04637},
  primaryClass = "cs.CR",
  keywords = {Computer Science - Cryptography and Security},
  year = 2018,
  month = apr,
  adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180404637A},
}
```

## Can I use this code to train MalConv on my own dataset?
The code provided is instructional and nonfunctional.  With a few minor changes, it can be made functional.  In particular, you must provide a URL to fetch file contents by sha256 hash.

## How does this MalConv model differ from that of Raff et al.?
 * Our model was trained on binary files from labeled samples in the EMBER training set.
 * The original paper used `batch_size = 256` and `SGD(lr=0.01, momentum=0.9, decay=UNDISCLOSED, nesterov=True )`.  We used
 `decay=1e-3` and `batch_size=100`.
 * It is unknown whether the original paper used a special symbol for padding.
 * The paper allowed for up to 2MB malware sizes, we use 1MB because of memory limits on a commonly-used Titan X.
