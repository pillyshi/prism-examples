```mermaid
flowchart TD
	node1["smdis_small_download"]
	node2["smdis_small_preprocess"]
	node3["smdis_small_split_folds"]
	node4["smdis_small_transform@0"]
	node5["smdis_small_transform@1"]
	node6["smdis_small_transform@2"]
	node7["smdis_small_transform@3"]
	node8["smdis_small_transform@4"]
	node1-->node2
	node2-->node3
	node3-->node4
	node3-->node5
	node3-->node6
	node3-->node7
	node3-->node8
```
