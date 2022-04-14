# gNetDGP
An End-to-End Graph Neural Network for Disease Gene Prioritization.

## Table of contents
* [Installation](#installation)
    * [Docker](#using-docker)
    * [Conda](#using-conda)
* [Usage](#usage)
    * [Train the generic model](#train-the-generic-model)
    * [Predict using the generic model](#predict-using-the-generic-model)
    * [Train the specific model](#train-the-specific-model)
    * [Predict using the specific model](#predict-using-the-specific-model)
* [Additional material](#additional-material)

## Installation
#### Using docker
We provide a [Dockerfile](Dockerfile) to setup a runtime. To use it run
```bash
docker build -t gNetDGP .
```
 
#### Using Conda
```bash
conda env create -f environment.yml
conda activate gnetdgp_env
```

## Usage
To get an overview of available commands use
```bash
python main.py --help
```

To list the available options on a specific command use
```bash
python main.py [COMMAND] --help
```

#### Train the generic model
To train a new generic model use
```bash
python main.py generic-train --training_data_path ./data/training/genes_diseases.tsv
```

For available options run
```bash
python main.py generic-train --help
```

#### Predict using the generic model
Provide an input file of gene, disease tuples 
like in the [test/example_input_generic.tsv](test/example_input_generic.tsv)

Then run the command
```bash
python main.py generic-predict test/example_input_generi.tsv
```
This will score the provided disease, gene tuples and return a augmented version of the input file with added scores.
The result is stored in `--out_file`, the default is `./generic_predict_results.tsv` 

The result is sorted by the predicted score by default. 
If you want to preserve the input order add the option `--sort_result_by_score False`

To use a specific pre-trained model for the prediction add the option `--model_path /path/to/model.ptm`.

To get a list of available genes in the model run
```bash
python main.py generic-predict --get_available_genes
```

To get a list of available diseases in the model run
```bash
python main.py generic-predict --get_available_diseases
```

#### Train the specific model
To train the specific model run
```bash
main.py specific-train --training_disease_genes_path ./data/training/genes_diseases.tsv --training_disease_class_assignments_path ./data/training/extracted_disease_class_assignments.tsv
```

For available options run
```bash
python main.py specific-train --help
```

#### Predict using the specific model
To predict disease scores for the specific mode, provide a `input_file` with entrez gene IDs (one per line) like e.g.
in [test/example_input_specific.tsv](test/example_input_specific.tsv). Use the `--model_path` flag 
to provide a pretrained specific model.

```bash
main.py specific-predict ./test/example_input_specific.tsv --model_path /path/to/choosen/pretrained/specific/model.ptm
```

This will score the provided gene IDs to be associated with the disease the model was trained on and return a augmented
version of the input file with added scores.

The result is stored in `--out_file`, the default is `./generic_predict_results.tsv` 

The result is sorted by the predicted score by default. 
If you want to preserve the input order add the option `--sort_result_by_score False`

## Additional material
The process of the reported experiments is documented in
* [results/disease_gene_prioritization.ipynb for the disease gene prioritization task.](results/disease_gene_prioritization.ipynb)
* [results/disease_gene_classification.ipynb for the disease type classification.](results/disease_gene_classification.ipynb)
