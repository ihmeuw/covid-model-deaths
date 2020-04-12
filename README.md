
covid-model-deaths
==================

A packaged version of the covid-19 deaths model.


### Setup.

To get setup to run the deaths model, do the following, 
get on the cluster in a qlogin and do the following:

```
> git clone https://stash.ihme.washington.edu/scm/cvd19/covid-model-deaths.git
> cd covid-model-deaths
> source setup_env.sh
```

Depending on how nice the cluster file system is, it will
take 1-5 minutes to setup a new conda environment and install
all the required libraries.  

After the command completes, you should already have the 
appropriate environment activated and you're ready to launch
a `jupyter lab` session and get started.

```
(covid-deaths-YYYY-MM-DD_HH-MM-SS) > jupyter lab .
```

`jupyter lab .` tells jupyter to open in the directory you're 
currently sitting in, which should be the `covid-model-deaths`
directory.  You can then navigate to `notebooks` sub-directory
and launch the notebooks for the model you're running.

TODO: Cleanup instructions/script
