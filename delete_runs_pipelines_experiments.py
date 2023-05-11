def delete_experiments():
    """
    Deletes all experiments in the Kubeflow Pipeline client.
    """
    import kfp

    # Connect to the Kubeflow Pipeline client
    client = kfp.Client()

    # Get a list of all experiments
    experiments = client.list_experiments()

    # Delete all runs in each experiment
    if experiments.experiments:
        for experiment in experiments.experiments:
            print(experiment.id)
            client.delete_experiment(experiment.id)

    return "Experiments deleted successfully."


def delete_runs():
    """
    Deletes all runs in the Kubeflow Pipeline client.
    """
    import kfp

    # Connect to the Kubeflow Pipeline client
    client = kfp.Client()

    # Get a list of all runs
    runs = client.list_runs()

    # Delete all runs
    if runs.runs:
        for run in runs.runs:
            print(run.id)
            kfp.Client().runs.delete_run(run.id)

    return "Runs deleted successfully."


def delete_pipelines():
    """
    Deletes all pipelines in the Kubeflow Pipeline client.
    """
    import kfp

    # Connect to the Kubeflow Pipeline client
    client = kfp.Client()

    # Get a list of all pipelines
    pipelines = client.list_pipelines()

    # Delete all pipelines
    if pipelines.pipelines:
        for pipeline in pipelines.pipelines:
            print(pipeline.id)
            kfp.Client().delete_pipeline(pipeline.id)

    return "Pipelines deleted successfully."
