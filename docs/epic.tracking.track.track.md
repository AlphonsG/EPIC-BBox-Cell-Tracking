<!-- markdownlint-disable -->

<a href="..\epic\tracking\track.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>function</kbd> `track`

```python
track(dirs: list[Path], config: dict[str, Any]) → None
```

Tracks objects in image sequences. 

Tracks objects in image sequences located in the given input directories using the object tracker specified in the EPIC YAML configuration file. Also saves and visualises corresponding object tracks in a subdirectory called 'Tracks' created in each input directory. 



**Args:**
 
 - <b>`dirs`</b>:  A sequence of existing input directories containing image sequences in common image formats (.png, .tiff, etc). 
 - <b>`config`</b>:  A loaded EPIC YAML configuration file. 


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
