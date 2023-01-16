<!-- markdownlint-disable -->

<a href="..\epic\detection\detect.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>function</kbd> `detect`

```python
detect(
    dirs: list[Path],
    config: dict[str, Any],
    detector: Model | None = None
) → list[Path]
```

Detects objects in images. 

Detects objects in images located in the given input directories using an object detector. If provided, uses the loaded object detector or, alternatively, loads and uses the detector specified in the EPIC YAML configuration file. Also saves and visualises corresponding object detections in a subdirectory called 'Detections' created in each input directory. 



**Args:**
 
 - <b>`dirs`</b>:  A sequence of existing input directories containing images  in common image formats (.png, .tiff, etc). 
 - <b>`config`</b>:  A loaded EPIC YAML configuration file. 
 - <b>`detector`</b>:  A loaded object detector to use. 



**Returns:**
 A sequence of paths to files specifying the object detections for each corresponding input directory. 


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
