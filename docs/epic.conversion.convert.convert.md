<!-- markdownlint-disable -->

<a href="..\epic\conversion\convert.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>function</kbd> `convert`

```python
convert(root_dir: Path, config: dict[str, Any]) → None
```

Converts datasets to an EPIC-compatible format. 

Converts datasets, in the form of directories that have a particular structure and contain images, to directories with a structure that can be processed by EPIC image processing commands. Does not modify input directories but instead produces outputs in an independent directory. 



**Args:**
 
 - <b>`root_dir`</b>:  A directory containing a dataset to convert. 
 - <b>`config`</b>:  A loaded EPIC YAML configuration file. 


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
