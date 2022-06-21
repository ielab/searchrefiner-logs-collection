# searchrefiner logs collection

This is a companion repository for the paper "The Impact of Query Refinement on Systematic Review Literature Search: A Query Log Analysis". This repo contains all the data and code used to produce the observations and figures in the paper. We don't provide the raw logs collected from the tool, since that contains identifiable data. However, this repo contains the code necessary to extract the raw logs into the same format as is contained here.

## Working with the data

The collection is split into three formats:

 - [searchrefiner.log.jsonl](searchrefiner.log.jsonl) contains just the processed log data, each query and metadata (i.e., seed studies, etc.) on a new line.
 - [searchrefiner.logical.log.jsonl](searchrefiner.logical.log.json) contains the log data grouped by logical sessions as described in the paper. The file starts with INVALID, which groups all queries that could not be grouped into a logical session.
 - [searchrefiner.mission.log.jsonl](searchrefiner.mission.log.json) contains the log data grouped by search missions as described in the paper. The file starts with INVALID, which groups all queries that could not be grouped into a search mission.

The code to produce the analysis and figures is located in [log_analysis.py](log_analysis.py). It is provided "as is" with no comments.

## Additional links

 - [searchrefiner repo](https://github.com/ielab/searchrefiner)
 - [searchrefiner demo](https://sr-accelerator.com/#/searchrefinery)
 - [read the paper](https://ielab.io/publications/scells-2022-searchrefiner-logs)

## Citing us

If you refer to or use this data in your paper, please use this citation:

```
@inproceedings{scells2022searchrefinerlogs,
    Author = {Scells, Harrisen and Forbes, Connor and Clark, Justin and Koopman, Bevan and Zuccon, Guido},
    Booktitle = {Proceedings of the 8th ACM SIGIR International Conference on the Theory of Information Retrieval},
    Organization = {ACM},
    Title = {The Impact of Query Refinement on Systematic Review Literature Search: A Query Log Analysis},
    Year = {2022}
}
```

If instead you wish to refer to the searchrefiner project:

```
@inproceedings{scells2018searchrefiner,
    Author = {Scells, Harrisen and Zuccon, Guido},
    Booktitle = {Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
    Organization = {ACM},
    Title = {searchrefiner: A Query Visualisation and Understanding Tool for Systematic Reviews},
    Year = {2018}
}
```