# DS301

The `.py` files can be ran on any devices wiith transformers and ray. To run the `.slurm` file, one needs to have access to [NYU HPC](https://www.nyu.edu/life/information-technology/research-and-data-support/high-performance-computing.html). After getting an NYU HPC account, just install the required environment package and change the username-related directory to the one of your own, then the script should be runable.

## Run python script (e.g. BERT wiith Bayes Optimization Search)
 ``` python bert_bayes.py BoolQ ```
 
## Run the slurm script (REMARK: Need access to NYU HPC!!)
``` sbatch bert_bayes.slurm ```

You can use
```watch -n 1 squeue -lu netid```

or load the log files (they are defined as `.err` and `.out` in the slurm script) to track the progress.
